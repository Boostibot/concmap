

#include <stdint.h>
#include <string.h>

#ifndef REQUIRE
    #include <assert.h>
    #define REQUIRE(x) assert(x)
    #define ASSERT(x) assert(x)
    #define TEST(x) assert(x)
#endif

#define XXHASH_FN64_PRIME_1  0x9E3779B185EBCA87ULL
#define XXHASH_FN64_PRIME_2  0xC2B2AE3D27D4EB4FULL
#define XXHASH_FN64_PRIME_3  0x165667B19E3779F9ULL
#define XXHASH_FN64_PRIME_4  0x85EBCA77C2B2AE63ULL
#define XXHASH_FN64_PRIME_5  0x27D4EB2F165667C5ULL

static inline uint64_t _xxhash64_rotate_left(uint64_t x, uint8_t bits)
{
    return (x << bits) | (x >> (64 - bits));
}

static inline uint64_t _xxhash64_process_single(uint64_t previous, uint64_t input)
{
    return _xxhash64_rotate_left(previous + input * XXHASH_FN64_PRIME_2, 31) * XXHASH_FN64_PRIME_1;
}

static uint64_t xxhash64(const void* key, int64_t size, uint64_t seed)
{
    uint32_t endian_check = 0x33221100;
    REQUIRE(*(uint8_t*) (void*) &endian_check == 0 && "Big endian machine detected! Please change this algorithm to suite your machine!");
    REQUIRE((key != NULL || size == 0) && size >= 0);

    uint8_t* data = (uint8_t*) (void*) key;
    uint8_t* end = data + size;
    
    //Bulk computation
    uint64_t hash = seed + XXHASH_FN64_PRIME_5;
    if (size >= 32)
    {
        uint64_t state[4] = {0};
        uint64_t block[4] = {0};
        state[0] = seed + XXHASH_FN64_PRIME_1 + XXHASH_FN64_PRIME_2;
        state[1] = seed + XXHASH_FN64_PRIME_2;
        state[2] = seed;
        state[3] = seed - XXHASH_FN64_PRIME_1;
        
        for(; data < end - 31; data += 32)
        {
            memcpy(block, data, 32);
            state[0] = _xxhash64_process_single(state[0], block[0]);
            state[1] = _xxhash64_process_single(state[1], block[1]);
            state[2] = _xxhash64_process_single(state[2], block[2]);
            state[3] = _xxhash64_process_single(state[3], block[3]);
        }

        hash = _xxhash64_rotate_left(state[0], 1)
            + _xxhash64_rotate_left(state[1], 7)
            + _xxhash64_rotate_left(state[2], 12)
            + _xxhash64_rotate_left(state[3], 18);
        hash = (hash ^ _xxhash64_process_single(0, state[0])) * XXHASH_FN64_PRIME_1 + XXHASH_FN64_PRIME_4;
        hash = (hash ^ _xxhash64_process_single(0, state[1])) * XXHASH_FN64_PRIME_1 + XXHASH_FN64_PRIME_4;
        hash = (hash ^ _xxhash64_process_single(0, state[2])) * XXHASH_FN64_PRIME_1 + XXHASH_FN64_PRIME_4;
        hash = (hash ^ _xxhash64_process_single(0, state[3])) * XXHASH_FN64_PRIME_1 + XXHASH_FN64_PRIME_4;
    }
    hash += (uint64_t) size;

    //Consume last <32 Bytes
    for (; data + 8 <= end; data += 8)
    {
        uint64_t read = 0; memcpy(&read, data, sizeof read);
        hash = _xxhash64_rotate_left(hash ^ _xxhash64_process_single(0, read), 27) * XXHASH_FN64_PRIME_1 + XXHASH_FN64_PRIME_4;
    }

    if (data + 4 <= end)
    {
        uint32_t read = 0; memcpy(&read, data, sizeof read);
        hash = _xxhash64_rotate_left(hash ^ read * XXHASH_FN64_PRIME_1, 23) * XXHASH_FN64_PRIME_2 + XXHASH_FN64_PRIME_3;
        data += 4;
    }

    while (data < end)
        hash = _xxhash64_rotate_left(hash ^ (*data++) * XXHASH_FN64_PRIME_5, 11) * XXHASH_FN64_PRIME_1;
        
    // Avalanche
    hash ^= hash >> 33;
    hash *= XXHASH_FN64_PRIME_2;
    hash ^= hash >> 29;
    hash *= XXHASH_FN64_PRIME_3;
    hash ^= hash >> 32;
    return hash;
}

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
        if(hashed <= REMOVED)
            hashed += 2;
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
    using Entry = Base::Entry;

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
    using Entry = Base::Entry;

    Base base = Base();
    uint64_t seed = 0;
    std::shared_mutex mutex;

    Entry* get(Key const& key)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        std::shared_lock<std::shared_mutex> lock(mutex);
        return map.get(key, hashed);
    }
    
    bool set(Key const& key, Value value)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        std::unique_lock<std::shared_mutex> lock(mutex);
        return map.set(key, std::move(value), hashed);
    }

    bool remove(Key const& key)
    {
        uint64_t hashed = Base::hash_escape(hash_func(key, seed));
        std::unique_lock<std::shared_mutex> lock(mutex);
        return map.remove(key, hashed);
    }
};

#include <vector>
template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Distributed_Map {
    using Base = Map_Base<Key, Value, equals>;
    using Entry = Base::Entry;

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
        entries.reserve(shard->map.size);
        for(size_t i = 0; i < shard->map.capacity; i++) {
            Entry* entry = &shard->map.data[i];
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
                
                std::shared_lock<std::shared_mutex> lock(shard->mutex);
                if(shard_gen->generation != shard->map.info_generation) {
                    shard_gen->generation != shard->map.info_generation; 
                    shard_gens->entries.clear();
                    push_all(&shard_gens->entries, &shard->map);
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
            shard->map.set(std::move(entry->key), std::move(entry->value), entry->hash);
        }
        unlock_all();
    }
};


inline static uint32_t current_thread_id() {
    return std::hash<std::thread::id>{}(std::this_thread::get_id());
}

struct Simple_EBR {
    struct alignas(std::hardware_destructive_interference_size) Slot {
        std::atomic<uint32_t> id;
    };

    Slot* data;
    uint32_t size; //always power of two

    Simple_EBR(uint32_t requested = std::thread::hardware_concurrency()*3/2){
        size = 1;
        while(size < requested)
            size *= 2;

        data = new Slot[size];
    } 

    ~Simple_EBR() {
        delete[] data;
    }

    struct Lock {
        uint32_t slot;
        uint32_t id;
        Simple_EBR* ebr;

        Lock(Simple_EBR* ebr, uint32_t thread_id = current_thread_id()) : ebr(ebr) {
            uint32_t mask = ebr->size - 1;
            uint32_t curr_slot = thread_id & mask;

            for(uint32_t iter = 1; ; iter++) {
                uint32_t curr_id = ebr->data[curr_slot].id.load();
                if(curr_id % 2 == 0) {
                    if(ebr->data[curr_slot].id.compare_exchange_strong(curr_id, curr_id + 1)) {
                        slot = curr_slot;
                        id = curr_id + 1;
                    }
                }

                curr_slot = (curr_slot + iter) & mask;
            }
        }

        ~Lock() {
            ebr->data[slot].id.store(id + 1);
        }

        void wait_for_others_to_leave() const 
        {
            for(uint32_t i = 0; i < ebr->size; i++) {
                uint32_t first = ebr->data[i].id.load();
                uint32_t curr = first;
                while(curr % 2 == 1 && curr == first) {
                    curr = ebr->data[i].id.load();
                    std::this_thread::yield();
                }
            }
            
            //enum {SMALL = 128};
            //uint32_t small[SMALL];
            //uint32_t* history = small;
            //if(ebr->size > SMALL)
            //    history = new uint32_t[ebr->size];

            //if(history != small)
            //    delete[] history;
        }
    };
};

struct Bag {
    struct Instance {
        std::atomic<Instance*> next;
        std::atomic<uint32_t> size;
        uint32_t capacity;
        uint32_t* data;
    };    

    Simple_EBR ebr;
    std::atomic<Instance*> current_instance;
    std::atomic<uint32_t> instance_generation;

    void insert(uint32_t data) {
        
        while(true)
        {
            Simple_EBR::Lock lock(&ebr);
            uint32_t instance_gen = instance_generation.load();
            Instance* instance = current_instance.load();

            if(instance->size + 1 > instance->capacity) {
                Instance new_isntance;
                
                instance_gen += 1;
            }
            
            uint32_t new_instance_gen = instance_generation.load();
            if(instance_gen == new_instance_gen)
                break;
        }
    }
};

template <typename T, size_t N>
struct Small_Array {
    T* data;
    uint32_t size;
    uint32_t capacity;
    alignas(T) uint8_t small[N*sizeof(T)];

    Small_Array(size_t cap = 0) {
        if(cap <= N)
            data = (T*) (void*) small;
        else 
            data = malloc(cap*sizeof(T));
        capacity = (uint32_t) cap;
    }

    ~Small_Array() {
        if(data != small)
            free(data);
    }

    void clear() {
        size = 0;
    }

    void push(T t) {
        if(size >= capacity) {
            ASSERT(data != small);
            capacity = capacity*3/2 + 8;
            data = realloc(data, capacity*sizeof(T));
        }

        data[size++] = t;
    }
};

template <typename Key, typename Value, bool equals(Key const&, Key const&), uint64_t hash_func(Key const& key, uint64_t seed)>
struct Atomic_Map {

    enum Status {
        EMPTY = 0,
        FULL = 1,
        GRAVESTONE = 2
        INSERTING = 3,
        MARKED_FOR_DELETE = 8 + 1
    };
    
    struct Slot {
        std::atomic<uint64_t> refs_status_hash; 
        uint32_t gen_counter; //max value is reserved
        Key key;
        Value value;
    };

    Slot* slots;
    uint32_t capacity;
    uint32_t count;

    bool get(Key const& key, uint32_t hash, Value* out) {
        struct History {
            uint32_t hash;
            uint32_t gen;
        };

        Small_Array<History, 64> histories[2];
        for(uint32_t repeat = 0; ; repeat++) {

            Small_Array<History, 64>& history = histories[repeat % 2];
            Small_Array<History, 64>& prev_history = histories[(repeat + 1) % 2];
            history.clear();

            uint32_t mask = capacity - 1;
            uint32_t i = hash & mask;
            for(uint32_t iter = 1;; iter++) {

                Slot* slot = &slots[i];
                while(true) {
                    //check if we are hasha and status matching 
                    //(this will get rid of 99% of all requests)   
                    uint64_t curr_ref_status_hash = slot->refs_status_hash.load();
                    uint32_t curr_status = (curr_ref_status_hash >> 32) & 0xF;
                    uint32_t curr_hash = curr_ref_status_hash & 0xFFFFFFFF;
                    if(curr_status == EMPTY)
                        goto exit_search;

                    if(curr_status != FULL || curr_hash != hash) {
                        history_push(&history, iter - 1, History{curr_hash, 0}, repeat);
                        break;
                    }

                    //try to mark this entry as being looked at. If fail go again
                    uint64_t new_ref_status_hash = curr_ref_status_hash + (uint64_t) 1 << 36;
                    if(slot->refs_status_hash.compare_exchange_strong(curr_ref_status_hash, new_ref_status_hash) == false)
                        continue;

                    uint32_t curr_gen = slot->gen_counter;
                    history.push(History{curr_hash, curr_gen});

                    //now noone will chnage key or value so we can compare key
                    //If we fail there was a hash collision (extremely rare)
                    if(equals(slot->key, key) == false)
                        break;

                    //else we can copy out the value 
                    *out = slot->value;
                    
                    //and mark the entry as not referenced anymore
                    uint64_t updated_ref_status_hash = slot->refs_status_hash.fetch_sub((uint64_t) 1 << 36) - (uint64_t) 1 << 36;

                    //if there was a remove operation that happened while we were looking at the key/value
                    // it has marked this slot as MARKED_FOR_DELETE. If we are the last one referencing a 
                    // MARKED_FOR_DELETE slot we should remove it by marking it as gravestone
                    uint32_t updated_status = (updated_ref_status_hash >> 32) & 0xF;
                    uint32_t updated_refs = updated_ref_status_hash >> 36;
                    if(updated_status == MARKED_FOR_DELETE && updated_refs == 0)
                        slot->refs_status_hash.store((uint64_t) GRAVESTONE << 32); 

                    return true;
                }
    
                i = (i + iter) & mask;
            }   

            exit_search:

            //check if we reached converged state. If we did we can be sure that the 
            if(repeat > 0 && history.size == prev_history.size) 
                if(memcmp(history.data, prev_history.data, history.size*sizeof *history.data) == 0)
                    return false;
        }

        return false;
    }

    bool remove(Key const& key, uint32_t hash, Value* out) {
        while(true) {
            //check if we are hasha and status matching 
            //(this will get rid of 99% of all requests)   
            uint64_t curr_ref_status_hash = refs_status_hash.load();
            uint32_t curr_status = (curr_ref_status_hash >> 32) & 0xF;
            uint32_t curr_hash = curr_ref_status_hash & 0xFFFFFFFF;
            if(curr_status != FULL || curr_hash != hash) 
                return false;

            //try to mark this entry as being looked at. If fail go again
            uint64_t new_ref_status_hash = curr_ref_status_hash + (uint64_t) 1 << 36;
            if(refs_status_hash.compare_exchange_strong(curr_ref_status_hash, new_ref_status_hash) == false)
                continue;

            //now noone will chnage key or value so we can compare key
            //If we fail there was a hash collision (extremely rare)
            if(equals(this->key, key) == false)
                return false;

            //else we can copy out the value 
            *out = this->value;

            //change the status to MARKED_FOR_DELETE and mark the entry as not referenced anymore
            //TODO: perform both as single atomic OP
            refs_status_hash.fetch_or((uint64_t) MARKED_FOR_DELETE << 32)
            uint64_t updated_ref_status_hash = refs_status_hash.fetch_sub((uint64_t) 1 << 36) - (uint64_t) 1 << 36;

            //if we are the last one remove it
            uint32_t updated_status = (updated_ref_status_hash >> 32) & 0xF;
            uint32_t updated_refs = updated_ref_status_hash >> 36;
            ASSERT(updated_status == MARKED_FOR_DELETE);
            if(updated_refs == 0)
                refs_status_hash.store((uint64_t) GRAVESTONE << 32); 

            return true;
        }

        return false;
    }
};

void test(Simple_EBR* ebr) {
    
    Simple_EBR::Lock lock(ebr);

    lock.wait_for_others_to_leave();

}

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

int main()
{
    Tracker_Node node;
    {
        auto t1 = Tracker<int>(1);
        auto t2 = Tracker<int>(1);

        t1 = t2;
        t2 = t1;

        auto t3 = t2;
    }

    node.dump();
    std::cout << "Hello World!\n";
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
