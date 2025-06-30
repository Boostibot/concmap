
#include <stdint.h>
#include <string.h>

#ifndef REQUIRE
    #include <assert.h>
    #define REQUIRE(x) assert(x)
    #define ASSERT(x) assert(x)
    #define TEST(x) assert(x)
#endif

#include <chrono>
static uint64_t clock_ns()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

static inline uint64_t random_splitmix(uint64_t* state) 
{
	uint64_t z = (*state += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

static inline uint64_t hash64_bijective(uint64_t x) 
{
    x = (x ^ (x >> 30)) * (uint64_t) 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * (uint64_t) 0x94d049bb133111eb;
    x = x ^ (x >> 31);
    return x;
}

template <typename Key, typename Value, bool equals(Key const&, Key const&)>
struct Map_Base {
    static constexpr uint64_t EMPTY = 0;
    static constexpr uint64_t REMOVED = 1;

    struct Entry{
        Key key;
        Value value;
        uint64_t hash;

        bool used() const {
            return hash != EMPTY && hash != REMOVED;
        }
    };
    
    Entry* data = 0;
    uint32_t size = 0;
    uint32_t capacity = 0; //always power of two
    uint32_t gravestone_count = 0;

    //Info fields - these are not used by the map itself
    //but are vital to evaluating the perf characteristics of it.
    //Generation is used by the multithreaded implementations
    uint32_t info_generation = 0; //gets incremented on every change
    uint32_t info_rehashes = 0; //gets incremented on every rehash
    uint32_t info_extra_probes = 0; //upper estimated to the number of extra probes needed to find every item in the map. Ideally would be 0


    Map_Base() = default;
    Map_Base(Map_Base const&) = delete;
    Map_Base& operator=(Map_Base const&) = delete;

    ~Map_Base() {
        for(uint64_t i = 0; i < capacity; i++) {
            Entry* entry = data[i];
            if(entry->hash > REMOVED)
                entry->~Entry();
        }

        free(data);
        memset(this, 0, sizeof *this);
    }

    Entry* get(Key const& key, uint64_t hashed) const
    {
        if(size > 0)
        {
            uint64_t mask = (uint64_t) capacity - 1;
            uint64_t index = hashed & mask; 
            for(uint64_t iter = 1;; iter ++) {
                Entry* entry = data[index];
                if(entry->hash == EMPTY)
                    break;
                
                if(entry->hash == hashed)
                    if(equals(entry->key, key))
                        return entry;
                
                ASSERT(iter <= capacity && "must not be completely full!");
                index = (index + iter) & mask;
            }
        }
        return NULL;
    }
    
    bool set(Key const& key, Value value, uint64_t hashed)
    {
        reserve((size_t) size + 1);
        info_generation += 1;

        uint64_t mask = (uint64_t) capacity - 1;
        uint64_t index = hashed & mask; 
        for(uint64_t iter = 1;; iter ++) {
            Entry* entry = data[index];
            if(entry->hash == EMPTY || entry->hash == REMOVED) {
                if(entry->hash == REMOVED)
                    info_extra_probes += iter - 1;

                new(entry) Entry{ 
                    key,
                    std::move(value),
                    hashed,
                };
                return false;
            }

            if(entry->hash == hashed && equals(entry->key, key)) {
                entry->value = std::move(value);
                return true;
            }
                
            ASSERT(iter <= capacity && "must not be completely full!");
            index = (index + iter) & mask;
        }
    }

    bool remove(Key const& key, uint64_t hashed)
    {
        Entry* entry = get(key, hashed);
        if(entry) {
            entry->~Entry();
            entry->hash = REMOVED;
            info_generation += 1;
        }

        return entry != NULL;
    }

    static uint64_t hash_escape(uint64_t hash) {
        if(hash <= REMOVED)
            hash += 2;
        return hash;
    }

    void reserve(size_t to_size)
    {
        if(capacity*3/4 <= to_size + gravestone_count)
            rehash(to_size);
    }
    
    void rehash(size_t to_cap)
    {
        //_hash_check_consistency(from_table);
        size_t required = gravestone_count + to_cap;
        if(gravestone_count > to_cap)
            required = to_cap;
          
        if(required < to_cap)
            required = to_cap;

        size_t new_capacity = 16;
        while(new_capacity*3/4 < required)
            new_capacity *= 2;

        TEST(new_capacity < UINT32_MAX);

        Entry* new_data = (Entry*) calloc(new_capacity, sizeof(Entry));
        size_t new_extra_probes = 0;

        uint64_t mask = (uint64_t) capacity - 1;
        for(size_t j = 0; j < capacity; j++)
        {
            Entry* entry = &data[j];
            if(entry->hash > REMOVED)
            {
                uint64_t i = entry->hash & mask;
                for(uint64_t it = 1;; it++) {
                    if(new_data[i].hash == EMPTY) {
                        new(&new_data[i]) Entry{ 
                            std::move(entry->key),
                            std::move(entry->value),
                            entry->hash,
                        };
                        break;
                    }

                    i = (i + it) & mask;
                    new_extra_probes += 1;
                }
                
                entry->~Entry();
            }
        }

        free(data);
        
        data = new_data;
        capacity = new_capacity;
        info_extra_probes = new_extra_probes;
        info_rehashes += 1;
    }

    void clear() {
    }
};

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <new>

template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Single_Map {
    using Base = Map_Base<Key, Value, equals>;
    using Entry = typename Base::Entry;

    Base base = Base();
    uint64_t seed = 0;
    
    Entry* get(Key const& key) const
    {
        return get(key, Base::hash_escape(hash_func(key, seed)));
    }
    
    bool set(Key const& key, Value value)
    {
        return set(key, std::move(value), Base::hash_escape(hash_func(key, seed)));
    }

    bool remove(Key const& key)
    {
        return remove(key, Base::hash_escape(hash_func(key, seed)));
    }
};

template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Mutex_Map {
    using Base = Map_Base<Key, Value, equals>;
    using Entry = typename Base::Entry;

    Base base = Base();
    uint64_t seed = 0;
    std::shared_mutex mutex;

    Entry* get(Key const& key)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        std::shared_lock<std::shared_mutex> lock(mutex);
        return base.get(key, hashed);
    }
    
    bool set(Key const& key, Value value)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        std::unique_lock<std::shared_mutex> lock(mutex);
        return base.set(key, std::move(value), hashed);
    }

    bool remove(Key const& key)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        std::unique_lock<std::shared_mutex> lock(mutex);
        return base.remove(key, hashed);
    }
};

#include <vector>
template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Distributed_Map {
    using Base = Map_Base<Key, Value, equals>;
    using Entry = typename Base::Entry;

    struct alignas(std::hardware_destructive_interference_size) Shard {
        std::shared_mutex mutex;
        Base map;
    };

    Shard* shards = NULL;
    size_t shards_count = 0; //always a power of two
    uint64_t seed = 0;
    
    explicit Distributed_Map(size_t min_num_shards = std::thread::hardware_concurrency()*3/2) {
        if(min_num_shards < 1) 
            min_num_shards = 1;

        shards_count = 1;
        while(shards_count < min_num_shards)
            shards_count *= 2;

        shards = new Shard[shards_count];
    }

    ~Distributed_Map() {
        delete[] shards;
    }

    Shard* get_shard(uint64_t hashed) const
    {
        uint64_t bits = 53;
        uint64_t rotl = (hashed << bits) | (hashed >> (64 - bits));
        return &shards[rotl & (shards_count - 1)];
    }

    Entry* get(Key const& key) const
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        Shard* shard = get_shard(hashed);

        std::shared_lock<std::shared_mutex> lock(shard->mutex);
        return shard->map.get(key, hashed);
    }
    
    bool set(Key const& key, Value value)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        Shard* shard = get_shard(hashed);

        std::unique_lock<std::shared_mutex> lock(shard->mutex);
        return shard->map.set(key, std::move(value), hashed);
    }

    bool remove(Key const& key)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        Shard* shard = get_shard(hashed);

        std::unique_lock<std::shared_mutex> lock(shard->mutex);
        return shard->map.remove(key, hashed);
    }
    
    //The remainder of the class is about implementing "global"/"across shard" functions.
    //There we need to get or enforce some property across multiple shards each potentially
    // concurrently accessed from multiple threads. We cannot simply iterate each shard and
    // get/enforce the property, as it could happen that by the time we move onto a different shard
    // some other thread changed the previous one. Simply put when empty() returns true than there
    // must have existed point in time (history more precisely) when there really were no entries 
    // simultaneously across all shards. 
    //One way to enforce this is to acquire locks of all shards
    // before doing anything. This is of course very inefficient and we try to avoid it. For this
    // we use the generation counters. Each action is performed in at least two iterations over the 
    // shards. First we iterate, lock, perform the action and save the generation. Second we iterate
    // again and check if there were any changes done in the meantime between the two iterations. If there
    // were not then there must have existed a time when the property was true. If not then we simply 
    // get/enforce again, until eventually the two consecutive iterations match.

    //small helper that avoid allocation of small vector in most cases 
    struct Generations {
        uint32_t data_inline[128];
        uint32_t* data; 

        Generations(size_t shards_count) {
            if(shards_count > 128)
                data = malloc(shards_count*sizeof(uint32_t));
            else
                data = data_inline;
        }

        ~Generations() {
            if(data != data_inline)
                free(data);
        }
    };

    void clear() 
    {
        Generations generations(shards_count);

        //iterate once and clear 
        for(size_t i = 0; i < shards_count; i++) {
            Shard* shard = &shards[i];
            
            std::unique_lock<std::shared_mutex> lock(shard->mutex);
            {
                shard->map.clear();
                generations.data[i] = shard->map.info_generation; 
            }
        }

        //iterate again and validate no changes were made. 
        // If there were then go again.
        for(size_t j = 0; ; j++) {
        
            bool had_changes = false;
            for(size_t i = 0; i < shards_count; i++) {
                Shard* shard = &shards[i];
            
                std::unique_lock<std::shared_mutex> lock(shard->mutex);
                {
                    if(generations.data[i] != shard->map.info_generation) {
                        shard->map.clear();
                        had_changes = true;
                        generations.data[i] = shard->map.info_generation;
                    }
                }
            }

            if(had_changes == false)
                break;
        }
    }
    
    size_t size() const 
    {
        Generations generations(shards_count);
         
        //iterate once and clear 
        for(size_t i = 0; i < shards_count; i++) {
            Shard* shard = &shards[i];
            
            std::shared_lock<std::shared_mutex> lock(shard->mutex);
            generations.data[i] = shard->map.size; 
        }
        
        //iterate again
        for(size_t j = 0; ; j++) {
            size_t sum = 0;
            bool had_changes = false;
            for(size_t i = 0; i < shards_count; i++) {
                Shard* shard = &shards[i];
                
                sum += generations.data[i];
                std::shared_lock<std::shared_mutex> lock(shard->mutex);
                if(generations.data[i] != shard->map.size) {
                    generations.data[i] = shard->map.size; 
                    had_changes = true;
                } 
            }

            if(had_changes == false)
                return sum;
        }
    }

    bool empty() const {
        return size() == 0;
    }

    static void push_all(std::vector<Entry>* entries, Base* base) {
        entries.reserve(base->size);
        for(size_t i = 0; i < base->capacity; i++) {
            Entry* entry = &base->data[i];
            if(entry->used())
                entries.push(*entry);
        }
    }

    std::vector<Entry> extract_entries() 
    {
        struct Shard_Gen {
            uint32_t generation;
            std::vector<Entry> entries;
        };    

        std::vector<Shard_Gen> shard_gens(shards_count);

        //iterate once and push all from given shard 
        for(size_t i = 0; i < shards_count; i++) {
            Shard* shard = &shards[i];
            Shard_Gen* shard_gen = &shard_gens[i];

            std::shared_lock<std::shared_mutex> lock(shard->mutex);
            shard_gen->generation = shard->map.info_generation;
            push_all(&shard_gen->entries, &shard->map);
        }
        
        //iterate again and override if generation change
        for(size_t j = 0; ; j++) {
            bool had_changes = false;
            for(size_t i = 0; i < shards_count; i++) {
                Shard* other_shard = &shards[i];
                Shard_Gen* shard_gen = &shard_gens[i];
                
                std::shared_lock<std::shared_mutex> lock(shard_gen->mutex);
                if(shard_gen->generation != shard_gen->map.info_generation) {
                    shard_gen->generation != shard_gen->map.info_generation; 
                    shard_gens->entries.clear();
                    push_all(&shard_gens->entries, &shard_gen->map);
                    had_changes = true;
                } 
            }

            if(had_changes == false)
                break;
        }

        //push all into a single vector
        std::vector<Entry> out;
        
        size_t combined_size = 0;
        for(size_t i = 0; i < shards_count; i++)
            combined_size += shard_gens[i].entries.size();
        out.reserve(combined_size);

        for(size_t i = 0; i < shards_count; i++) {
            Shard_Gen* shard_gen = &shard_gens[i];
            for(size_t j = 0; j < shard_gen->entries.size(); j++)
                out.push_back(std::move(shard_gen->entries[j]));
        }

        return out;
    }
    
    void lock_all() {
    
    }
    
    void unlock_all() {
    
    }

    //It might be faster to just lock everything briefly.
    //The extraction of entries might take quite a bit of time 
    // so if there is another thread modifying the map fast enough
    // we might never converge.
    std::vector<Entry> extract_entries_locked() 
    {
        std::vector<Entry> entries;
        lock_all();

        for(size_t i = 0; i < shards_count; i++) 
            push_all(&entries, &shards[i].map);

        unlock_all();
        return entries;
    }
    
    void copy(Distributed_Map const& other) {
        std::vector<Entry> entries = other.extract_entries(); 
        lock_all();
        for(size_t i = 0; i < shards_count; i++) 
            shards[i].map.clear();
        
        for(size_t i = 0; i < entries.size(); i++) {
            Entry* entry = &entries[i];
            shards[i].map.set(std::move(entry->key), std::move(entry->value), entry->hash);
        }
        unlock_all();
    }
};

template <typename T, size_t N>
struct Small_Array {
    T* data;
    uint32_t size;
    uint32_t capacity;
    alignas(T) uint8_t small[N*sizeof(T)];

    Small_Array(size_t cap = 0) {
        capacity = N;
        reserve(cap);
    }

    ~Small_Array() {
        if(data != small)
            free(data);
    }

    void clear() {
        size = 0;
    }

    void reserve(uint32_t to_cap) {
        if(to_cap > capacity) {
            capacity = capacity*3/2 + 8;
            if(capacity < to_cap)
                capacity = to_cap;

            if(data == small) {
                data = malloc(capacity*sizeof(T));
                memcpy(data, small, size*sizeof(T));
            }
            else {
                data = realloc(data, capacity*sizeof(T));
            }
        }
    }

    void push(T t) {
        reserve(size + 1);
        data[size++] = t;
    }
};

template <typename T>
struct Gen_Ptr {
    std::atomic<uint64_t> val;
    
    static_assert(sizeof(T*) == 8);
    static constexpr uint64_t MULT = ((uint64_t) 1 << 48) / alignof(T);
    static constexpr uint64_t MAX_GEN = ((uint64_t) 1 << 16) * alignof(T);
    
    struct Decompressed {
        T* ptr;
        uint64_t gen;
        uint64_t val;

        Gen_Ptr<T> encode() const {
            return Gen_Ptr{Gen_Ptr::encode(ptr, gen)};
        }
    };

    static Decompressed decode(uint64_t val) {
        uint64_t mult = ((uint64_t) 1 << 48) / alignof(T);
        uint64_t gen = val / mult;
        uint64_t ptr_num = val % mult;
        uint64_t ptr = ptr_num * alignof(T);

        //or high bits of ptr to fix up the pointer pattern
        //by taking bits of the stack pointer. This is achieved by taking adress
        // of some dummy thats removed by the optimizer.
        int dummy = 0; dummy = dummy + 0;

        ptr |= ((uint64_t) &dummy) & ((uint64_t) (-1) << 48);
        return Decompressed{(T*) ptr, gen, val};
    }

    static uint64_t encode(T* ptr, uint64_t gen) {
        uint64_t ptr_num = (uint64_t) ptr / alignof(T);
        uint64_t mult = ((uint64_t) 1 << 48) / alignof(T);
        uint64_t gen_num = gen * mult;
        uint64_t out = gen_num | ptr_num;
        return out;
    }
    
    static Gen_Ptr tick_up(uint64_t val, uint64_t by = 1) {
        uint64_t mult = ((uint64_t) 1 << 48) / alignof(T);
        return val + mult;
    }

    Decompressed decode() const {
        return decode(val);
    }
    
    Gen_Ptr tick_up(uint64_t by = 1) const {
        uint64_t mult = ((uint64_t) 1 << 48) / alignof(T);
        return val + mult;
    }
    
    bool cas(uint64_t old_val, T* ptr) {
        //uint64_t old_val = val.load(std::memory_order_relaxed);
        Decompressed decoded = decode();
        uint64_t new_val = encode(ptr, decoded.gen + 1);
        return val.compare_exchange_strong(old_val, new_val);
    }

    void set(T* ptr) {
        Decompressed decoded = decode(val.load(std::memory_order_relaxed));
        val = encode(ptr, decoded.gen + 1);
    }
    
    T* get() const {
        return decode(val).ptr;
    }
};

struct alignas(std::hardware_destructive_interference_size) Distrint {
    std::atomic<uint64_t> gen_val;
};

void distrint_add(Distrint* distributed, uint32_t N, uint64_t val, uint32_t at, uint32_t value_bits) {
    
    assert((N & (N - 1)) == 0, "must be power of two!");
    uint32_t i = at & (N - 1);

    //add (1, val) to (gen, stored)
    uint64_t added = 1ull << value_bits | val;
    distributed[i].gen_val.fetch_add(added);
}

void distrint_sub(Distrint* distributed, uint32_t N, uint64_t val, uint32_t at, uint32_t value_bits) {
    
    assert((N & (N - 1)) == 0, "must be power of two!");
    uint32_t i = at & (N - 1);
    uint64_t gen_mask = (uint64_t) -1 << value_bits;
    
    //add (0, -val) to (gen, stored)
    //where -val is the complement. This complement causes
    //overflow in stored which in turn ticks up gen counter
    uint64_t added = (-val) & ~gen_mask;
    distributed[i].gen_val.fetch_add(added);
}

uint64_t distrint_get(Distrint* distributed, uint32_t N, uint32_t value_bits) 
{
    uint64_t gen_mask = (uint64_t) -1 << value_bits;
    if(N == 1)
        return distributed[0].gen_val & ~gen_mask;

    enum {MAX = 64};
    assert((N & (N - 1)) == 0, "must be power of two!");
    assert(N < MAX);

    uint64_t history[MAX]; (void) history; //not initialized
    for(uint32_t i = 0; i < N; i++) 
        history[i] = distributed[i].gen_val.load(std::memory_order_relaxed);
        
    for(uint32_t repeat = 0;; repeat++) {
        uint64_t sum = 0;

        bool all_same = true;
        for(uint32_t i = 0; i < N; i++) {
            uint64_t val = distributed[i].gen_val.load(std::memory_order_relaxed);
            sum += val;

            if(val != history[i]) {
                history[i] = val;
                all_same = false;
            }
        }

        if(all_same) {
            return sum & ~gen_mask;
        }
    }
}

template <uint32_t N, uint32_t value_bits = 32>
struct Distributed_Int {
    Distrint distributed[N];

    void add(uint64_t val, uint32_t at) { distrint_add(distributed, N, val, at, value_bits); }
    void sub(uint64_t val, uint32_t at) { distrint_sub(distributed, N, val, at, value_bits); }
    uint64_t get() const                { distrint_get(distributed, N, value_bits); }
};

template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Fine_Lock_Map 
{
    enum {MIN_CAPACITY = 64};

    struct Node {
        std::atomic<Node*> next;
        uint64_t hash;
        Key key;
        Value value;
    };
    
    struct Slot {
        std::shared_mutex mutex;
        std::atomic<Node*> first;
    };

    struct Table {
        uint64_t capacity;
        std::atomic<uint32_t> count; //can be distributed eventually
        std::atomic<uint32_t> migrated;

        alignas(std::hardware_destructive_interference_size)
        Slot slots[];
    };

    mutable std::atomic<uint32_t> head_tail_index;
    mutable std::atomic<Table*> tables[32];

    Fine_Lock_Map(uint32_t initial_capacity = MIN_CAPACITY) {
        
    }

    void migrate(uint32_t head, uint32_t tail, uint32_t max_times = 3) const
    {
        //attempt to migrate all entries from a single slot.
        //If the given slot is empty tries up to max_times.
        Table* head_table = tables[head].load(std::memory_order_relaxed);
        Table* tail_table = tables[tail].load(std::memory_order_relaxed);
        for(uint32_t rep = 0; rep < max_times; rep) {
            uint64_t migrated = head_table->migrated.fetch_add(1);
            uint64_t capacity = 1ull << i;
            Slot* slot = &table->slots[migrated];

            //if we have already migrated everything quit
            if(migrated >= capacity) 
                return;

            //double locking of the migrated bucket
            Node* first = table[migrated].first.load(std::memory_order_relaxed);
            if(first == NULL)
                continue;
            
            //attempt to migrate all entries
            {
                std::shared_lock lock(table[migrated].mutex);
                first = table[migrated].first.load(std::memory_order_relaxed);
                for(Node* curr = first; curr != NULL; ) {
                    Node* next = curr->next.load(std::memory_order_relaxed);
                
                    //transplant node into the newer table. 
                    //If is not found there, inserts it
                    // else we simply delete it
                    bool found = set_or_transplant_into_table(tail_table, curr->key, curr->hash, NULL, curr);
                    if(found == false)
                        delete curr;
                    curr = next;
                }
                table[migrated].first.store(NULL, std::memory_order_relaxed);
            }

            //If was the last one bump up the head table
            if(i == capacity - 1) {
                uint32_t curr_head_tail = head_tail_index.fetch_add(1 << 16);
                uint32_t curr_head = curr_head_tail >> 16;
                uint32_t curr_tail = curr_head_tail & 0xFFFF;
                ASSERT(curr_head == head);
                ASSERT(curr_tail >= tail);
            }

            break;
        }
    }

    bool set_or_transplant_into_table(Table* table, uint32_t table_i, Key const& key, uint64_t hash, Value* value, Node* old_node) {
        uint64_t capacity = 1ull << i;
        uint64_t i = hash & (capacity - 1);
        Slot* slot = &table->slots[i];

        slot->mutex.lock();
        Node* first = slot->first.load(std::memory_order_relaxed);
        for(Node* curr = first; curr != NULL; curr = curr->next.load(std::memory_order_relaxed)) {
            if(curr->hash == hash && equals(curr->key, key)) {
                if(old_node == NULL)
                    curr->value = std::move(*value);

                slot->mutex.unlock();
                return true;
            }
        }

        Node* new_node = old_node ? old_node : new Node(first, hash, key, std::move(value));
        slot->first.store(new_node, std::memory_order_relaxed);
        slot->mutex.unlock();

        //increase the count and if too much allocate a new bigger table
        uint32_t count = table->count.fetch_add(1);
        if(count > capacity*3/4) 
            try_create_new_table(table_i);

        return false;
    }
    
    bool try_create_new_table(uint32_t table_i) {
        ASSERT(table_i <= 31);

        //check once again if nobody has allocate a bigger table (just in case)
        uint32_t curr_head_tail = head_tail_index.load();
        uint32_t curr_tail = curr_head_tail & 0xFFFF;
        if(table_i == curr_tail && tables[table_i + 1].load() == NULL) {

            //allocate the bigger table and try to claim a spot
            uint64_t capacity = table_i*2;
            Table* new_table = malloc(sizeof(Table) + capacity*sizeof(Slot));
            Table* null_table = NULL;
            if(tables[new_tail + 1].compare_exchange_strong(null_table, new_table)) {
                    
                //properly initialize the new table
                new_table->count = 0;
                new_table->migrated = 0;
                for(uint32_t k = 0; k < new_table; k++)
                    new (new_table->slots + k) Slot();

                //publish it
                uint32_t new_head_tail = head_tail_index.fetch_add(1);
                uint32_t new_tail = new_head_tail & 0xFFFF;
                ASSERT(new_tail == curr_tail);
                return true;
            }
            else {
                free(new_table);
            }
        }
        return false;
    }

    void set(Key const& key, uint64_t hash, Value value) {
        uint32_t head_tail = head_tail_index.load(std::memory_order_relaxed);
        uint32_t head = head_tail >> 16;
        uint32_t tail = head_tail & 0xFFFF;

        Table* table = tables[tail].load(std::memory_order_relaxed);
        set_or_transplant_into_table(table, key, hash, &value, NULL);
        
        if(head != tail)
            migrate(head, tail);
    }

    bool get(Key const& key, uint64_t hash, Value* out) const {
        uint64_t head_tail = head_tail_index.load();
        uint32_t head = head_tail >> 16;
        uint32_t tail = head_tail & 0xFFFF;
        
        //go thorugh the tables newest to oldest
        // and look for a single entry matching key. 
        //Each table can in theory contain a different entry with the same key
        // so we just return the newest. Restart if newest table has changed
        // (yes we have to restart the whole thing since it could happen
        // that an entry was migrated just in between us switching tables thus
        // it was there yet we didnt register it)
        for(uint32_t rep = 0; ; rep++) {

            for(uint32_t table_i = tail; table_i >= head; table_i--) {
                Slot* table = tables[table_i].load();
                uint64_t i = hash & (table->capacity - 1);
                    
                //double locking
                Node* first = table[i].first.load(std::memory_order_relaxed);
                if(first == NULL)
                    continue;

                std::shared_lock lock(table[i].mutex);
                first = table[i].first.load(std::memory_order_relaxed);
                for(Node* curr = first; curr != NULL; curr = curr->next.load(std::memory_order_relaxed)) {
                    if(curr->hash == hash && equals(curr->key, key)) {
                        if(out)
                            *out = curr->value;
                    
                        if(head != tail)
                            migrate(head, tail);
                        return true;
                    }
                }
            }
        
            uint64_t new_head_tail = head_tail_index.load();
            uint32_t new_head = new_head_tail >> 16;
            uint32_t new_tail = new_head_tail & 0xFFFF;
            if(tail == new_tail)
                break;

            tail = new_tail;
            head = new_head;
        }

        if(head != tail)
            migrate(head, tail);
        return false;
    }

    bool remove(Key const& key, uint64_t hash, Value* out) {
        bool found = false;
        
        uint64_t head_tail = head_tail_index.load(std::memory_order_relaxed);
        uint32_t head = head_tail >> 16;
        uint32_t tail = head_tail & 0xFFFF;
        
        //go thorugh the tables oldest to newest
        // and delete all entries with the given key. Reload the tail
        // after each iter so that we end on really the newest table
        for(uint32_t table_i = head; table_i <= tail; table_i++) {
            Slot* table = tables[table_i].load();
            uint64_t i = hash & (table->capacity - 1);
                    
            //double locking
            Node* first = table[i].first.load(std::memory_order_relaxed);
            if(first != NULL)
            {
                std::unique_lock lock(table[i].mutex);
                first = table[i].first.load(std::memory_order_relaxed);
            
                //find node and dele from chain
                std::atomic<Node*>* prev = &table[i].first;
                for(Node* curr = first; curr != NULL;) {
                    Node* next = curr->next.load(std::memory_order_relaxed);
                    if(curr->hash == hash && equals(curr->key, key)) {
                        prev->store(next, std::memory_order_relaxed);
                        found = true;
                        if(out)
                            *out = std::move(curr->value);

                        delete curr;
                        break;
                    }

                    prev = &curr->next;
                    curr = next;
                }
            }
                
            head_tail = head_tail_index.load(std::memory_order_relaxed);
            head = head_tail >> 16;
            tail = head_tail & 0xFFFF;
        }
            
        if(head != tail)
            migrate(head, tail);

        return found;
    }
};

//#include "spmc_queue.h"

struct EBR_OLD {
    struct alignas(std::hardware_destructive_interference_size) Slot {
        std::atomic<uint64_t> gen_and_taken;
        std::atomic<uint64_t> thread_id;

        //more or less arbitrary per "thread" payload
        std::atomic<uint64_t> hash;

    };

    struct Instance {
        uint64_t capacity;
        Instance* prev;
        Slot* slots[];
    };
    
    std::atomic<Instance*> instance;

    EBR_OLD(size_t initial_capacity = 16) {
        if(initial_capacity <= 0)
           initial_capacity = 1; 


        TEST(initial_capacity < UINT64_MAX);
        instance = make_instance(NULL, (uint64_t) initial_capacity);
    }

    ~EBR_OLD() {
        Instance* first = this->instance.load();
        for(Instance* curr = first; curr != NULL; )
        {
            Instance* prev = curr->prev;
            delete_instance(curr);
            curr = prev;
        }
    }

    Slot* lock(uint64_t hash, uint32_t thread_id) {
        for(uint64_t j = 0; ; j++) {
            Instance* instance = this->instance.load();
            ASSERT(instance);

            uint64_t mask = instance->capacity - 1;
            uint64_t i = thread_id & mask;
            for(uint64_t iter = 1; iter <= mask; i = (i + 1) & mask) {
                Slot* slot = instance->slots[i];
                uint64_t gen_and_taken = slot->gen_and_taken;

                //look for not taken slot
                if(gen_and_taken % 2 == 0)
                    //take it
                    if(slot->gen_and_taken.compare_exchange_strong(gen_and_taken, gen_and_taken | 1)) {
                        gen_and_taken |= 1;

                        slot->hash = hash;
                        slot->thread_id = thread_id;

                        //mark as full
                        slot->gen_and_taken = gen_and_taken + 2;
                    }
            }
            
            //If we iterated all slots at least once, the instance hasnt changed and we still didnt find
            // an empty slot, allocate a bigger instance. This new instance contains the same slots
            // as the old one (we jst copy *pointers* not the contents) but also has some new slots.
            //Note that ABA cannot happen since we never delete or abandon instances in use.
            if(j > 0 && instance == this->instance.load()) {
                Instance* new_instance = make_instance(instance, instance->capacity*2);
                if(this->instance.compare_exchange_strong(instance, new_instance) == false) {
                    //if someone made a new instance in the meantime, delete and try again...
                    delete_instance(new_instance);
                }
            }
        }
    }

    void unlock(Slot* slot) {
        //clear the taken bit
        uint64_t gen_and_taken = slot->gen_and_taken.load(std::memory_order_relaxed);
        slot->gen_and_taken = gen_and_taken & ~(uint64_t) 1;
    }
    
    static Instance* make_instance(Instance* prev, uint64_t requested_capacity) {
        uint64_t capacity = prev ? prev->capacity : 1;
        if(capacity < 1)
            capacity = 1;

        while(capacity < requested_capacity)
            capacity *= 2;

        Instance* new_instance = (Instance*) calloc(1, sizeof(Instance) + capacity*sizeof(Slot*));
        new_instance->capacity = capacity;
        new_instance->prev = prev;

        if(prev) {
            for(uint64_t i = 0; i < prev->capacity; i++) 
                new_instance->slots[i] = prev->slots[i];
        }

        for(uint64_t i = prev ? prev->capacity : 0; i < capacity; i++) 
            new_instance->slots[i] = new Slot;

        return new_instance;
    }

    static void delete_instance(Instance* inst) {
        for(uint32_t i = inst->prev ? inst->prev->capacity : 0; i < inst->capacity; i++) 
            delete inst->slots[i];
        
        free(inst);
    }
    
    struct Lock {
        EBR_OLD* ebr;
        Slot* slot;

        Lock(EBR_OLD& ebr, uint64_t hash, uint32_t thread_id) {
            this->ebr = &ebr;
            this->slot = ebr.lock(hash, thread_id);
        }

        ~Lock() {
            ebr->unlock(slot);
        }
    };
};

struct EBR {
    static constexpr uint32_t EMPTY_THREAD_ID = 0;
    typedef void* (*Create_Node_Func)(void* context);
    typedef void (*Publish_Node_Func)(void* node, void* context);
    typedef void (*Delete_Node_Func)(void* node, void* context);
    
    struct EBR_Config {
        size_t node_ebr_offset = 0;
        Create_Node_Func create_node_func = NULL;
        Publish_Node_Func publish_node_func = NULL; //optional. Gets called after a node was sucessfully published in a single ebr instance
        Delete_Node_Func delete_node_func = NULL;
        void* func_context = NULL;
        double collect_every_s = 0.016; //16ms = one frame
    };
    
    struct EBR_Node_Data {
        std::atomic<uint64_t> gen_and_taken;
        std::atomic<uint64_t> last_access_clock_ns;
        std::atomic<uint32_t> thread_id;
    };

    struct Slot {
        std::atomic<uint32_t> id;
        std::atomic<EBR_Node_Data*> data;
    };

    struct Instance {
        uint64_t capacity;
        Instance* prev;
        Slot slots[];
    };
    
    std::atomic<Instance*> instance;

    EBR(size_t initial_capacity = 32) {
        instance = make_instance(NULL, initial_capacity);
    }

    ~EBR() {
        Instance* first = this->instance.load();
        for(Instance* curr = first; curr != NULL; )
        {
            Instance* prev = curr->prev;
            delete_instance(curr);
            curr = prev;
        }
    }

    void node_state(EBR_Config const& config, void* user_data) {
        EBR_Node_Data* data = (EBR_Node_Data*) ((uint8_t*) user_data + config.node_ebr_offset);

        uint32_t thread_id = 0;
        uint64_t last_access = 0;

        uint64_t gen0 = data->gen_and_taken.load(std::memory_order_acquire);
        while(true) {
            thread_id = data->thread_id.load(std::memory_order_relaxed);
            last_access = data->last_access_clock_ns.load(std::memory_order_relaxed);
            uint64_t gen1 = data->gen_and_taken.load(std::memory_order_acquire);

            if(gen0 == gen1)
                break;
            
            gen0 = gen1;
        }

        
    }

    void* lock(EBR_Config const& config, uint32_t thread_id = current_thread_id()) {
        for(uint64_t retry = 0; ; retry++) {
            Instance* instance = this->instance.load();
            ASSERT(instance);

            uint64_t mask = instance->capacity - 1;
            uint64_t i = thread_id & mask;
            for(uint64_t iter = 0; iter <= mask; iter++, i = (i + iter) & mask) {
                Slot* slot = &instance->slots[i];
                
                uint32_t id = slot->id.load(std::memory_order_relaxed);
                if(id == thread_id) {
                    //slot->id => slot->data and we can use relaxed since
                    // if slot->id == thread_id we were the thread that did 
                    // the storing so that relaxed loads are ordered (from our POV).
                    //
                    //BUT! someone else could have overriden this slot thus we dont have to
                    // be the thread that originally created it. Thats fine though since
                    // data is immutable, thus it will always be there.
                    EBR_Node_Data* data = slot->data.load(std::memory_order_relaxed);
                    ASSERT(data);
                    
                    //Even though we are the rightful owners, in the time between 
                    // the slot->id == thread_id and here, we could have been scheduled out.
                    //Thus it is entirely possible for this threads slot to be in in the meantime replaced
                    // by some other thread. We need to ensure that only one thread is doing the storing
                    // of the data to ensure integrity. Thus we CAS fight over the taken flag
                    uint64_t gen_and_taken = data->gen_and_taken.load(std::memory_order_relaxed);
                    if(gen_and_taken % 4 == 0)
                        if(data->gen_and_taken.compare_exchange_strong(gen_and_taken, gen_and_taken + 1, 
                            std::memory_order_acquire, std::memory_order_relaxed
                        )) {
                            printf("TID %08llx found itself in slot #%lli\n", (long long) thread_id, (long long) i);
                            data->last_access_clock_ns.store(clock_ns(), std::memory_order_relaxed);
                            data->thread_id.store(thread_id, std::memory_order_relaxed);

                            //if accidentally stole this slot then repair it
                            // (can happen when slot->id == thread_id succeeds, then we go to sleep,
                            // another thread replaces us, does full lock + unlock sequence, then we wake
                            // up and also successfully CAS gen_and_taken, yet by this point the thread id
                            // differs).
                            if(slot->id.load(std::memory_order_relaxed) != thread_id)
                                slot->id.store(thread_id, std::memory_order_relaxed);


                            //mark as full
                            data->gen_and_taken.store(gen_and_taken + 2, std::memory_order_release);
                            printf("TID %08llx locked in slot #%lli\n", (long long) thread_id, (long long) i);

                            void* user_data = (uint8_t*) data - config.node_ebr_offset;
                            return user_data;
                        }
                }
            }
            
            //we didnt find ourselves => slow path
            printf("TID %08llx didnt find anything... attempting to insert\n", (long long) thread_id);

            //because the above for loop can take a long time
            // reload instance, mask etc (better to stay relevant than have to redo work later on)
            instance = this->instance.load();
            mask = instance->capacity - 1;
            uint64_t max_distance = mask/4; //we want to be able find our slot relatively quickly
            
            int64_t collect_every_ns = (int64_t) (config.collect_every_s*1e9);

            //try to add oneself to an empty slot or replace some slots that are:
            // 1) in EMPTY state
            // 2) accessed more than collect_every_ns ago
            i = thread_id & mask;
            for(uint64_t iter = 0; iter <= max_distance; iter++, i = (i + iter) & mask) {
                Slot* slot = &instance->slots[i];

                uint32_t id = slot->id.load();
                EBR_Node_Data* data = slot->data.load();
                if(data == NULL) {

                    void* user_data = config.create_node_func(config.func_context);

                    EBR_Node_Data* new_data = (EBR_Node_Data*) ((uint8_t*) user_data + config.node_ebr_offset);
                    new_data->thread_id.store(EMPTY_THREAD_ID, std::memory_order_relaxed);
                    if(slot->data.compare_exchange_strong(data, new_data) == false)
                        config.delete_node_func(user_data, config.func_context);
                    else if(config.publish_node_func)
                        config.publish_node_func(user_data, config.func_context);

                    data = slot->data.load();
                    printf("TID %08llx helped allocate EBR_Node_Data 0x%p\n", (long long) thread_id, data);
                }
                ASSERT(data);
                
                uint64_t now = clock_ns();
                uint64_t gen_and_taken = data->gen_and_taken.load();
                uint64_t last_access_clock_ns = data->last_access_clock_ns.load();
                int64_t ago = (int64_t) (now - last_access_clock_ns);

                if(gen_and_taken % 4 == 0 && (id == EMPTY_THREAD_ID || ago > collect_every_ns)) {
                    if(data->gen_and_taken.compare_exchange_strong(gen_and_taken, gen_and_taken + 1)) {
                        if(id == EMPTY_THREAD_ID)
                            printf("TID %08llx placed in an empty slot #%lli\n", (long long) thread_id, (long long) i);
                        else
                            printf("TID %08llx replaced last thread %08llx last used %es ago in slot #%lli\n", (long long) thread_id, (long long) data->thread_id, ago*1e-9, (long long) i);
                            

                        data->last_access_clock_ns.store(clock_ns());
                        data->thread_id.store(thread_id);
                        slot->id.store(thread_id);

                        //mark as full
                        data->gen_and_taken.store(gen_and_taken + 2);
                        
                        void* user_data = (uint8_t*) data - config.node_ebr_offset;
                        return user_data;
                    }
                }
            }

            //If we iterated all slots at least once, the instance hasnt changed and we still didnt find
            // an empty slot, allocate a bigger instance. This new instance contains the same EBR_Node_Data
            // as the old one (we jst copy pointers) but also has some new slots.
            //All slots (even the ones that point to the old EBR_Node_Data) dont have their TIDs set, thus threads
            // will fight again over the slots.
            //Note that ABA cannot happen since we never delete or abandon instances in use.
            if(instance == this->instance.load()) {
                printf("TID %08llx making instance of size %lli\n", (long long) thread_id, (long long) instance->capacity*2);
                Instance* new_instance = make_instance(instance, instance->capacity*2);
                if(this->instance.compare_exchange_strong(instance, new_instance) == false) {
                    //if someone made a new instance in the meantime, delete and try again...
                    delete_instance(new_instance);
                    printf("TID %08llx making instance of size %lli failed... retrying lock %llix\n", (long long) thread_id, (long long) instance->capacity*2, (long long) retry);
                }
                else {
                    printf("TID %08llx making instance of size %lli success... retrying lock %llix\n", (long long) thread_id, (long long) instance->capacity*2, (long long) retry);
                }
            }
        }
    }

    void unlock(EBR_Config const& config, void* user_data) 
    {
        EBR_Node_Data* data = (EBR_Node_Data*) ((uint8_t*) user_data + config.node_ebr_offset);

        uint32_t tid = data->thread_id.load(std::memory_order_relaxed);
        printf("TID %08llx unlocked\n", (long long) tid);
        
        //clear the taken bit
        uint64_t gen_and_taken = data->gen_and_taken.load(std::memory_order_relaxed);
        ASSERT(gen_and_taken % 4 == 2);
        data->gen_and_taken.store(gen_and_taken + 2, std::memory_order_relaxed);
    }
    
    Instance* make_instance(Instance* prev, uint64_t requested_capacity) 
    {
        uint64_t capacity = prev ? prev->capacity : 1;
        if(capacity < 1)
            capacity = 1;

        while(capacity < requested_capacity)
            capacity *= 2;

        Instance* new_instance = (Instance*) calloc(1, sizeof(Instance) + capacity*sizeof(Slot));
        TEST(new_instance != NULL);
        new_instance->capacity = capacity;
        new_instance->prev = prev;
        if(prev) {
            for(uint64_t i = 0; i < prev->capacity; i++) {
                new_instance->slots[i].id.store(EMPTY_THREAD_ID, std::memory_order_relaxed);
                new_instance->slots[i].data.store(prev->slots[i].data.load(std::memory_order_relaxed), std::memory_order_relaxed);
            }
        }

        for(uint64_t i = prev ? prev->capacity : 0; i < capacity; i++) {
            new_instance->slots[i].id.store(EMPTY_THREAD_ID, std::memory_order_relaxed);
            new_instance->slots[i].data.store(NULL, std::memory_order_relaxed);
        }

        return new_instance;
    }
    
    void delete_instance(Instance* inst) {
        //if(delete_node_func != NULL)
        //    for(uint64_t i = inst->prev ? inst->prev->capacity : 0; i < inst->capacity; i++)  {
        //        EBR_Node_Data* data = inst->slots[i].data.load(std::memory_order_relaxed);
        //        void* user_data = (uint8_t*) data - node_ebr_offset;
        //        delete_node_func(user_data, func_context);
        //    }

        free(inst);
    }

    struct Lock {
        EBR* ebr;
        void* data;
        const EBR_Config* config;

        Lock(EBR& ebr, EBR_Config const* config, uint32_t thread_id = current_thread_id()) {
            this->ebr = &ebr;
            this->data = ebr.lock(*config, thread_id);
            this->config = config;
        }

        ~Lock() {
            ebr->unlock(*config, data);
        }
    };
    
    
    struct Private_Lock {
        EBR* ebr;
        EBR_Node_Data* data;
        uint64_t gen_and_taken;

        Private_Lock(EBR& ebr, EBR_Config const& config, uint32_t thread_id = current_thread_id()) {
            this->ebr = &ebr;
            this->data = ebr.lock(thread_id);
        }

        ~Private_Lock() {
            ebr->unlock(data);
        }
    };

    uint64_t private_lock(EBR_Node_Data* data, uint64_t clock_val = clock_ns(), uint32_t thread_id = current_thread_id()) {
        uint64_t gen_and_taken = data->gen_and_taken.load(std::memory_order_relaxed);
        data->gen_and_taken.store(gen_and_taken + 1, std::memory_order_relaxed);

        data->last_access_clock_ns.store(clock_val, std::memory_order_relaxed);
        data->thread_id.store(thread_id, std::memory_order_relaxed);

        data->gen_and_taken.store(gen_and_taken + 2, std::memory_order_release);
        return gen_and_taken + 2;
    }

    void private_unlock(EBR_Node_Data* data, uint64_t gen_and_taken) {
        ASSERT(gen_and_taken % 4 == 2);
        data->gen_and_taken.store(gen_and_taken + 2, std::memory_order_relaxed);
    }
};


struct Collection {
    struct Thread_Private;

    EBR ebr;
    Thread_Private* thread_data;
    uint32_t ref_count;
    
    struct Node {
        Node* next;
        int data;
    };
    struct Thread_Private {
        EBR::EBR_Node_Data ebr_data;
        Collection* central;
        Thread_Private* next;
        Node* removed_nodes;
        bool is_private;
        bool is_deleted;
    };
    
    void cleanup(Collection* central, Thread_Private* self) {
        std::shared_ptr<int> p;
    }
    
    static void* create_thread_private(void* context)
    {
        return new Thread_Private();
    }
    
    static void delete_thread_private(void* node, void* context)
    {
        delete (Thread_Private*) node;
    }
    
    static void publish_thread_private(void* node, void* context)
    {
        Thread_Private* tprivate = (Thread_Private*) node;
        Collection* central = (Collection*) context;
        tprivate->next = central->thread_data; 
        central->thread_data = tprivate;
    }
};


template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Atomic_Map2 {

    struct Node {
        Gen_Ptr<Node*> next; 
        uint64_t hash;
        Key key;
        Value value;
    };

    Gen_Ptr<Node*>* buckets;
    uint64_t capacity;
    EBR ebr;

    bool get(Key const& key, uint64_t hash, Value* out) const {
        EBR::Lock lock(ebr, hash, 0);

        //TODO: repeat with instances...

        uint64_t bucket_i = hash & (capacity - 1);
        Gen_Ptr<Node*>* bucket = &buckets[bucket_i];

        //do classic multiple repeats
        uint64_t prev_bucket_gen = (uint64_t) -1;
        for(uint64_t rep = 0; ; rep++) {
            Gen_Ptr<Node*> curr = *bucket;

            //There were no insertions iff gen counter hasnt moved
            // thus if two generations are equal we know that nothing new was added
            uint64_t curr_gen = curr.decode().gen;
            if(curr_gen == prev_bucket_gen) //&& instance == instance...
                break;

            //iterate all nodes in this bucket
            for(uint64_t iter = 0; ; iter++) {
                typename Gen_Ptr<Node*>::Decompressed curr_d = curr.decode();
                Node* node = curr_d.ptr;
                if(node == NULL) 
                    break;

                if(node->hash == hash && equals(node->key, key))
                {
                    if(out)
                        *out = node->value;
                    return true;
                }

                curr = curr_d.next;
            }
            
            prev_bucket_gen = curr_gen;
        }

        return false;
    }
    
    bool set(Key const& key, uint64_t hash, Value value) {
        Node* new_node = new Node(0, hash, key, std::move(value));

        EBR::Lock lock(ebr, hash, 0);

        uint64_t bucket_i = hash & (capacity - 1);
        Gen_Ptr<Node*>* bucket = &buckets[bucket_i];
        
        typename Gen_Ptr<Node*>::Decompressed curr_bucket;
        while(false) {
            //NOTE: this is a ginat mess! better to unroll everything at this point!
            curr_bucket = bucket->decode();
            new_node->next.set(curr_bucket.ptr);
            if(bucket->cas(curr_bucket.val, new_node))
                break;
        }

        //iterate nodes after and delete any that match the key
        bool had = remove_from(&new_node->next, curr_bucket.ptr, key, hash);
        return had;
    }

    bool remove(Key const& key, uint64_t hash) {
        uint64_t bucket_i = hash & (capacity - 1);
        Gen_Ptr<Node*>* bucket = &buckets[bucket_i];
        return remove_from(bucket, bucket->decode().ptr, key, hash);
    }

    bool remove_from(Gen_Ptr<Node*>* first_prev, Node* first_node, Key const& key, uint64_t hash) {
        struct Rem {
            Gen_Ptr<Node*>* prev;
            Node* node;
            Node* next;
            uint64_t prev_val;
        };
        
        uint64_t deleted = 0;
        Small_Array<Rem, 16> to_delete;
        while(true) {
            Gen_Ptr<Node*>* prev = first_prev;
            Node* node = first_node;

            for(uint64_t iter = 0; node != NULL; iter++) {
                Node* next = node->next.decode().ptr;

                if(node->hash == hash && equals(node->key, key))
                    to_delete.push(Rem{prev, node, next});

                prev = &node->next;
                node = next;
            }
        
            //NOTE: is all_ok necessary? 
            // -> it largely doesnt matter since its so rare for two deletes to race
            bool all_ok = true;
            for(uint32_t i = to_delete.size; i-- > 0;) {
                Rem pair = to_delete.data[i];
                if(pair.prev->cas(pair.prev_val, pair.next)) {
                    
                    deleted += 1;
                }
                else
                    all_ok = false;
            }

            if(all_ok)
                break;
                
            to_delete.clear();
        }

        return deleted > 0;
    }
};


struct Tracker_Node;
static std::atomic<Tracker_Node*> _current_installed_tracker;

#include <iostream>
struct Tracker_Node {

    Tracker_Node* next = NULL;
    std::atomic<int64_t> created = 0;
    std::atomic<int64_t> destroyed = 0;
    std::atomic<int64_t> copy_constructors = 0;
    std::atomic<int64_t> move_constructors = 0;
    std::atomic<int64_t> copy_assignments = 0;
    std::atomic<int64_t> move_assignments = 0;
    std::atomic<int64_t> comparisons = 0;
    std::atomic<uint64_t> id = 0;
    bool details = false;
    
    static Tracker_Node* current() {
        return _current_installed_tracker.load();
    }

    Tracker_Node(bool details = true) {
        next = _current_installed_tracker;
        _current_installed_tracker = this;
        this->details = details;
    }

    ~Tracker_Node() {
        TEST(created == destroyed);
        _current_installed_tracker = next;
    }

    void dump(const char* name = NULL)
    {
        if(name)
            std::cout << name << " (Tracker):\n";
        std::cout << "created: " << created << std::endl;
        std::cout << "destroyed: " << destroyed << std::endl;
        std::cout << "destroyed: " << destroyed << std::endl;
        std::cout << "copy_constructors: " << copy_constructors << std::endl;
        std::cout << "move_constructors: " << move_constructors << std::endl;
        std::cout << "copy_assignments: " << copy_assignments << std::endl;
        std::cout << "move_assignments: " << move_assignments << std::endl;
        std::cout << "comparisons: " << comparisons << std::endl;
    }
};

template<typename T>
struct Tracker {
    Tracker_Node* node = NULL;
    int64_t id = 0;
    T value;
    
    Tracker(T&& value, Tracker_Node* node) : value(std::move(value)) {
        assert(node);
        this->node = node;
        id = node->created.fetch_add(1);
    }
    
    Tracker(T&& value = T()) : value(std::move(value)) {
        Tracker_Node* node = Tracker_Node::current();
        assert(node);
        this->node = node;
        id = node->created.fetch_add(1);
    }

    Tracker(Tracker const& other) : value(other.value) {
        node = other.node;
        id = node->created.fetch_add(1);
        if(node->details) 
            node->copy_constructors += 1;
    }
    
    Tracker(Tracker const&& other) : value(std::move(other.value)) {
        node = other.node;
        id = node->created.fetch_add(1);
        if(node->details) 
            node->move_constructors += 1;
    }

    ~Tracker() {
        node->destroyed += 1;
    }

    Tracker& operator=(Tracker const& other) {
        value = other.value;
        if(node->details)
            node->copy_assignments += 1;
        return *this;
    }

    Tracker& operator=(Tracker const&& other) {
        value = std::move(other.value);
        if(node->details)
            node->move_assignments += 1;
        return *this;
    }

    bool operator==(Tracker const& other) const {
        if(node->details)
            node->comparisons += 1;

        return value == other.value;
    }
    bool operator!=(Tracker const& other) const {
        if(node->details)
            node->comparisons += 1;

        return value != other.value;
    }
};

template<typename T>
static inline uint64_t raii_tracker_hash(Tracker<T> const& tracker, uint64_t seed)
{

}

void test_gen_ptr()
{
    int a = 0; a = a + 0;
    int b = 1; b = b + 0;

    Gen_Ptr<int> ptr = {};
    ptr.set(&a);
    ASSERT(ptr.decode().ptr == &a);

    Gen_Ptr<int> roundtrip = ptr.decode().encode();
    ASSERT(ptr.val == roundtrip.val);
}

void test_tracker() {

    Tracker_Node node;
    {
        auto t1 = Tracker<int>(1);
        auto t2 = Tracker<int>(1);

        t1 = t2;
        t2 = t1;

        auto t3 = t2;
    }
    node.dump();
}

void test_ebr()
{
    EBR ebr(1e9, 1);


    {
        EBR::Lock lock(ebr, 1);
    }
    
    {
        EBR::Lock lock(ebr, 2);
    }
    
    {
        ebr.collect_every_ns = 0;
        EBR::Lock lock(ebr, 33);
        EBR::Lock lock1(ebr, 1);
        EBR::Lock lock2(ebr, 2);
    }
}

int main()
{
    test_gen_ptr();
    test_tracker();
    test_ebr();

    std::cout << "All done";
}

#endif

#include "benchmarks.h"

int main() {
    bench_all();
}