#pragma once

#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <new>
#include <chrono>

#include <assert.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t clock_ns()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

static inline void delay(uint64_t ns) {

    uint64_t curr = clock_ns();
    while(curr + ns > clock_ns());
}

static inline uint32_t hash32(uint32_t x) 
{
    x = ((x >> 16) ^ x) * 0x119de1f3;
    x = ((x >> 16) ^ x) * 0x119de1f3;
    x = (x >> 16) ^ x;
    return x;
}

static inline uint32_t unhash32(uint32_t x) 
{
    x = ((x >> 16) ^ x) * 0x119de1f3;
    x = ((x >> 16) ^ x) * 0x119de1f3;
    x = (x >> 16) ^ x;
    return x;
}

static inline uint64_t hash64(uint64_t x) 
{
    x = (x ^ (x >> 30)) * (uint64_t) 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * (uint64_t) 0x94d049bb133111eb;
    x = x ^ (x >> 31);
    return x;
}

static inline uint64_t unhash64(uint64_t x) 
{
    x = (x ^ (x >> 31) ^ (x >> 62)) * (uint64_t) 0x319642b2d24d8ec3;
    x = (x ^ (x >> 27) ^ (x >> 54)) * (uint64_t) 0x96de1b173f119089;
    x = x ^ (x >> 30) ^ (x >> 60);
    return x;
}

static inline uint32_t artificial_thread_id()
{
    static std::atomic<uint32_t> counter = 0;
    static thread_local uint32_t tid = counter.fetch_add(1);
    return tid;
}

static inline uint64_t random_splitmix64() 
{
    static thread_local uint64_t state = (uint64_t) &state + clock_ns();
	uint64_t z = (state += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}
 

struct Distributed {
    enum {MAX = 64};

    struct alignas(std::hardware_destructive_interference_size) Single {
        std::atomic<uint32_t> val;
    };

    Single data[MAX];
    uint32_t count;

    inline Distributed(uint32_t requested) {
        count = 1;
        while(count < requested && count < MAX)
            count *= 2;
    }

    inline void add(uint32_t val, uint32_t at) {
        uint32_t i = at & (count - 1);
        data[i].val.fetch_add(val);
    }

    inline uint32_t load_at(uint32_t at) const {
        uint32_t i = at & (count - 1);
        return data[i].val;
    }

    inline uint32_t load() const {
        
        uint32_t history[MAX]; (void) history; //not initialized
        for(uint32_t i = 0; i < count; i++) 
            history[i] = data[i].val.load(std::memory_order_relaxed);

        for(uint32_t repeat = 0;; repeat++) {
            uint32_t sum = 0;

            bool all_same = true;
            for(uint32_t i = 0; i < count; i++) {
                uint32_t val = data[i].val.load(std::memory_order_relaxed);
                sum += val;

                if(sum != history[i]) {
                    history[i] = val;
                    all_same = false;
                }
            }

            if(all_same)
                return sum;
        }
    }
};

#include <vector>
struct Bench_Raw_Result {
    uint64_t iters;
    uint64_t okays;
    uint64_t duration_ns;
};

static inline void dummy_bench_func(uint64_t*, uint64_t*){}

template<typename Fn>
void _bench_thread_func(std::atomic<uint64_t>* control, std::atomic<uint64_t>* begin, std::atomic<uint64_t>* end, Bench_Raw_Result* result, std::atomic<uint64_t>* dummy_sum, Fn const& func)
{
    while(true) {
        begin->fetch_add(1);

        while(true) {
            uint64_t cur = control->load();
            if(cur == (uint64_t) -1)
                goto exit;
            if(cur % 2 != 0)
                break;
        }

        uint64_t iters = 0;
        uint64_t okays = 0;
        uint64_t dummy = 0;

        uint64_t before = clock_ns();
        for(; control->load(std::memory_order_relaxed) % 2 == 1 || iters == 0; iters++)
            func(&okays, &dummy);
    
        uint64_t after = clock_ns();

        result->iters = iters;
        result->okays = okays;
        result->duration_ns = after - before;
        dummy_sum->fetch_add(dummy);

        end->fetch_add(1);
    }
    
    exit:
    end->fetch_add(1);
}

template<typename Fn1, typename Fn2>
std::vector<std::vector<Bench_Raw_Result>> lunch_bench_threads(double seconds, size_t trial_count, Fn1 const& func1, size_t count1, Fn2 const& func2, size_t count2, double cooldown = 0)
{
    size_t count = count1 + count2;
    std::vector<std::vector<Bench_Raw_Result>> column_results;
    std::atomic<uint64_t> begin = 0;
    std::atomic<uint64_t> end = 0;
    std::atomic<uint64_t> control = 0;

    std::vector<Bench_Raw_Result> results(count);
    static std::atomic<uint64_t> dummy_sum = 0;

    for(size_t i = 0; i < count1; i++) {
        Bench_Raw_Result* res = &results[i];
        std::thread t([&, res]{_bench_thread_func<Fn1>(&control, &begin, &end, res, &dummy_sum, func1);});
        t.detach();
    }
    
    for(size_t i = 0; i < count2; i++) {
        Bench_Raw_Result* res = &results[i + count1];
        std::thread t([&, res]{_bench_thread_func<Fn2>(&control, &begin, &end, res, &dummy_sum, func2);});
        t.detach();
    }
        
    //we repeat (trial) the function multiple times for efficiency.
    //The big part of the expense part of this function is the thread creation. 
    // By repeating just this loop we armortize against it...
    for(size_t trial_i = 1; trial_i <= trial_count; trial_i++) {
        //wait for all to start successfully
        while(begin != trial_i*count)
            std::this_thread::yield();
    
        control += 1; //set start
        std::this_thread::sleep_for(std::chrono::microseconds((long long) (1e6*seconds/trial_count))); //sleep while threads do their thing
        control += 1; //set end

        //wait for all to end successfully
        while(end != trial_i*count)
            std::this_thread::yield();

        column_results.push_back(results);

        for(size_t i = 0; i < count; i++)
            assert(results[i].iters > 0);

        if(cooldown > 0)
            std::this_thread::sleep_for(std::chrono::microseconds((long long) (1e6*cooldown)));
    }

    //end all trials and wait for threads to leave
    control = (uint64_t) -1;
    while(end != (trial_count + 1)*count)
        std::this_thread::yield();

    //transpose to be more friendly to process
    std::vector<std::vector<Bench_Raw_Result>> all_results(count, std::vector<Bench_Raw_Result>(trial_count));
    for(size_t i = 0; i < count; i++) 
        for(size_t trial = 0; trial < trial_count; trial ++)
            all_results[i][trial] = column_results[trial][i];

    return all_results;
}

#include <string>
#include <ostream>
#include <fstream>
#include <filesystem>
#include <iostream>

void bench_raw_result_to_csv_comment(std::ostream& stream)
{
    stream << "\"first line in chunk: [name, number_of_threads, number_of_trials], next lines: [iters, successful_iters, duration_in_ns] repeating number_of_trials times. This then repeats.\"\n";
}
void bench_raw_result_to_csv(std::ostream& stream, std::vector<std::vector<Bench_Raw_Result>> const& results, const char* name)
{
    size_t trials = results.size() ? results[0].size() : 0;
    stream << "\"" << name << "\", " << results.size() << ", " << trials << "\n";
    for(std::vector<Bench_Raw_Result> const& trials : results) {
        for(size_t i = 0; i < trials.size(); i++) {
            Bench_Raw_Result const& result = trials[i];
            if(i != 0)
                stream << ", ";

            stream << result.iters << ", " << result.okays << ", " << result.duration_ns;
        }
        stream << "\n";
    }
}

template<typename Fn1, typename Fn2>
bool bench_process_and_output_csv(double time, size_t trials, size_t min_threads, size_t max_threads, size_t thread_incr, const char* folder, const char* filename, const char* name, Fn1 const& func1, Fn2 const& func2, bool antisym = false) {
    
    size_t num_runs = 0;
    for(size_t i = min_threads; ; i += thread_incr) {
        if(i >= max_threads)
           i = max_threads; 
        num_runs += 1;
        if(i >= max_threads)
            break;
    }
    double time_per_thread = time / num_runs;

    const char* sym = antisym ? "mix" : "sym";

    printf("\n");
    printf("===============================================\n");
    printf("Starting bench=%s trials=%i time=%lfs time/thread=%lfs %s\n", name, (int) trials, time, time_per_thread, sym);
    printf("===============================================\n");

    std::string filename_complete = filename 
        ? std::string(folder) + "/" + filename
        : std::string(folder) + "/" + name + ".csv";
    std::filesystem::create_directories(std::filesystem::path(filename_complete).parent_path());
    bool exists = std::filesystem::exists(filename_complete);

    std::ofstream file_raw(filename_complete, std::ios::binary | std::ios::app);
    if(file_raw.is_open() == false) 
        printf("failed to open file %s\n", filename_complete.data());

    if(!exists)
        bench_raw_result_to_csv_comment(file_raw);

    for(size_t i = min_threads; ; i += thread_incr) {
        if(i >= max_threads)
           i = max_threads; 
     
        num_runs += 1;
        if(antisym == false)
        {
            printf("threads=%i\n", (int) i);
            std::vector<std::vector<Bench_Raw_Result>> raw = lunch_bench_threads<Fn1, decltype(dummy_bench_func)>(time_per_thread, (size_t) trials, func1, (size_t) i, dummy_bench_func, 0);
            bench_raw_result_to_csv(file_raw, raw, name);
        }
        else if(i > 1)
        {
            printf("threads=%i\n", (int) i);
            std::vector<std::vector<Bench_Raw_Result>> raw = lunch_bench_threads<Fn1, Fn2>(time_per_thread, (size_t) trials, func1, 1, func2, i - 1);
            bench_raw_result_to_csv(file_raw, raw, name);
        }

        if(i >= max_threads)
            break;
    }

    return file_raw.is_open();
}

void bench_all() {
    uint64_t non_atomic_target = 0;
    std::atomic<uint64_t> target = 0;
    std::mutex mutex;
    std::shared_mutex shared_mutex;

    auto load = [&](uint64_t* okay, uint64_t* dummy){
        dummy += target.load(std::memory_order_relaxed);
        *okay += 1;
    };
    
    auto store = [&](uint64_t* okay, uint64_t* dummy){
        target.store(0, std::memory_order_relaxed);
        *okay += 1;
    };
    
    auto faa = [&](uint64_t* okay, uint64_t* dummy){
        target.fetch_add(1, std::memory_order_relaxed);
        *okay += 1;
    };
    
    auto _and = [&](uint64_t* okay, uint64_t* dummy){
        target.fetch_and(1, std::memory_order_relaxed);
        *okay += 1;
    };
    
    auto cas = [&](uint64_t* okay, uint64_t* dummy){
        uint64_t expected = 0;
        target.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        *okay += 1;
    };
    
    auto load_cas = [&](uint64_t* okay, uint64_t* dummy){
        uint64_t expected = target.load(std::memory_order_relaxed);
        *okay += target.compare_exchange_strong(expected, expected + 1, std::memory_order_relaxed);
    };
    
    auto lock = [&](uint64_t* okay, uint64_t* dummy){
        mutex.lock();
        non_atomic_target += 1;
        mutex.unlock();
        *okay += 1;
    };

    auto read_lock = [&](uint64_t* okay, uint64_t* dummy){
        shared_mutex.lock_shared();
        *dummy += non_atomic_target;
        shared_mutex.unlock_shared();
        *okay += 1;
    };
    
    auto write_lock = [&](uint64_t* okay, uint64_t* dummy){
        shared_mutex.lock();
        non_atomic_target += 1;
        shared_mutex.unlock();
        *okay += 1;
    };

    auto faa_delay000 = [&](uint64_t* okay, uint64_t* dummy){
        target.fetch_add(1, std::memory_order_relaxed);
        delay(0);
        *okay += 1;
    };

    auto faa_delay100 = [&](uint64_t* okay, uint64_t* dummy){
        target.fetch_add(1, std::memory_order_relaxed);
        delay(100);
        *okay += 1;
    };
    
    auto faa_delay200 = [&](uint64_t* okay, uint64_t* dummy){
        target.fetch_add(1, std::memory_order_relaxed);
        delay(200);
        *okay += 1;
    };
    
    auto faa_delay400 = [&](uint64_t* okay, uint64_t* dummy){
        target.fetch_add(1, std::memory_order_relaxed);
        delay(400);
        *okay += 1;
    };

    double time = 4;
    size_t trials = 10;
    size_t min_threads = 1;
    size_t max_threads = (size_t) std::thread::hardware_concurrency();
    size_t incr_threads = 1;
    const char* folder = "bench_data";
    const char* file = "bench.csv";

    std::filesystem::remove(std::filesystem::path(folder) / file);

    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "load", load, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "store", store, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "faa", faa, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "and", _and, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "cas_all", cas, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "cas_success", load_cas, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "lock", lock, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "read_lock", read_lock, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "write_lock", write_lock, dummy_bench_func);
    
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "faa_delay000", faa_delay000, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "faa_delay100", faa_delay100, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "faa_delay200", faa_delay200, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "faa_delay400", faa_delay400, dummy_bench_func);

    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "store_mix_load", store, load, true);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "faa_mix_load", faa, load, true);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "and_mix_load", _and, load, true);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "cas_mix_load", cas, load, true);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "load_cas_mix_load", load_cas, load, true);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file, "write_lock_mix_load", write_lock, read_lock, true);
}