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

#ifdef _MSC_VER
    #include "intrin.h"
    static inline bool atomic_bit_test_and_set(std::atomic<uint32_t>* val, uint32_t offset)
    {
        return _interlockedbittestandset((long*) (void*) val, offset);
    }

    static inline bool atomic_bit_test_and_reset(std::atomic<uint32_t>* val, uint32_t offset)
    {
        return _interlockedbittestandreset((long*) (void*) val, offset);
    }
#elif defined(__GNUC__) || defined(__clang__) 
    static inline bool atomic_bit_test_and_set(std::atomic<uint32_t>* val, uint32_t offset)
    {
        uint32_t val = 1 << bit;
        return (a->fetch_or(val) & val) != 0;
    }

    static inline bool atomic_bit_test_and_reset(std::atomic<uint32_t>* val, uint32_t offset)
    {
        uint32_t val = 1 << bit;
        return (a->fetch_and(~val) & val) != 0;
    }
#else
    #error bad compiler
#endif

#define CHAN_ARCH_UNKNOWN   0
#define CHAN_ARCH_X86       1
#define CHAN_ARCH_X64       2
#define CHAN_ARCH_ARM32     3
#define CHAN_ARCH_ARM64     4

#ifndef CHAN_ARCH
    #if defined(_M_CEE_PURE) || defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
        #define CHAN_ARCH CHAN_ARCH_X86
    #elif defined(__x86_64__) || defined(_M_X64) || defined(__amd64__) && !defined(_M_ARM64EC) 
        #define CHAN_ARCH CHAN_ARCH_X64
    #elif defined(_M_ARM64) || defined(_M_ARM64EC) || defined(__aarch64__) || defined(__ARM_ARCH_ISA_A64)
        #define CHAN_ARCH CHAN_ARCH_ARM64
    #elif defined(_M_ARM32) || defined(_M_ARM32EC) || defined(__arm__) || defined(__ARM_ARCH)
        #define CHAN_ARCH CHAN_ARCH_ARM32
    #else
        #define CHAN_ARCH CHAN_ARCH_UNKNOWN
    #endif
#endif

#ifndef CHAN_CUSTOM_PAUSE
    #ifdef _MSC_VER
        #include <intrin.h>
        #if CHAN_ARCH == CHAN_ARCH_X86 || CHAN_ARCH == CHAN_ARCH_X64
            #define _CHAN_PAUSE_IMPL() _mm_pause()
        #elif CHAN_ARCH == CHAN_ARCH_ARM
            #define _CHAN_PAUSE_IMPL() __yield()
        #endif
    #elif defined(__GNUC__) || defined(__clang__) 
        #if CHAN_ARCH == CHAN_ARCH_X86 || CHAN_ARCH == CHAN_ARCH_X64
            #include <x86intrin.h>
            #define _CHAN_PAUSE_IMPL() _mm_pause()
        #elif CHAN_ARCH == CHAN_ARCH_ARM64
            #define _CHAN_PAUSE_IMPL() asm volatile("yield")
        #endif
    #endif

    static inline void pause_instruction() { 
        #ifdef _CHAN_PAUSE_IMPL
            _CHAN_PAUSE_IMPL();
        #endif
    } 
#endif

static inline void delay(uint64_t ns) {

    uint64_t curr = clock_ns();
    while(curr + ns > clock_ns())
        for(int i = 0; i < 10; i++)
            pause_instruction();
}

static inline void pause(uint64_t times) {

    for(uint64_t i = 0; i < times; i++)
        pause_instruction();
}


#include <vector>
#include <condition_variable>
#include <mutex>
struct Bench_Result {
    uint64_t iters;
    uint64_t okays;
    uint64_t duration_ns;
};

struct _Bench_Thread_Control {
    alignas(std::hardware_destructive_interference_size)
    std::atomic<uint64_t> state;
    std::atomic<uint64_t> begin;
    std::atomic<uint64_t> end;
    std::atomic<uint64_t> dummy_sum;
    std::mutex waiting_mutex;
    std::condition_variable waiting;

    alignas(std::hardware_destructive_interference_size)
    uint64_t threads;
    uint64_t cycle; 

    uint64_t start_time;
    uint64_t end_time;
};

static inline void dummy_bench_func(uint64_t*, uint64_t*){}

//The benchmarking strategy used below stems from the following factors:
// 1 - when the benchmark is running the benchmark orchestrator thread needs to sleep.
//     Because of this we cannot have it wait, then wake up and tell the benchmark to stop (simplest solution).
//     That would result in there being some time when its woken up yet the benchmark is still running.
//     This in turn produces slowdown visible in the result when all cores are used 
//     (since then the orchestrator cannot simply be moved to another core and not fight over the timeslice of other threads).
// 2 - The benchmark is run repeatedly in so called "trials". Only the median trial is then used for the final result. 
//     We could simply call the benchmark repeatedly however this amortizes the overhead of thread creation, thus allows
//     us to be done slightly faster with the whole benchmark suite.
template<typename Fn>
void _bench_thread_func(_Bench_Thread_Control* control, Bench_Result* result, Fn const& func)
{
    auto notify_started = [&]{
        control->waiting_mutex.lock();

        uint64_t begun = control->begin.fetch_add(1);
        if((begun + 1) % control->threads == 0) 
            control->waiting.notify_all();

        control->waiting_mutex.unlock();
    };
    auto notify_stopped = [&]{
        control->waiting_mutex.lock();

        uint64_t ended = control->end.fetch_add(1);
        if((ended + 1) % control->threads == 0) 
            control->waiting.notify_all();

        control->waiting_mutex.unlock();
    };

    while(true) {
        notify_started();
        
        uint64_t state = 0;
        while(true) {
            state = control->state.load();
            if(state == (uint64_t) -1) {
                notify_stopped();
                return;
            }
            if(state % 2 != 0)
                break;

            pause(10);
        }
        
        uint64_t iters = 0;
        uint64_t okays = 0;
        uint64_t dummy = 0;
        uint64_t start_time = control->start_time;
        uint64_t end_time = control->end_time;
        uint64_t cycle = control->cycle;

        //wait for start time
        while((int64_t) (clock_ns() - start_time) < 0)
            pause(10);

        uint64_t before = clock_ns();
        {
            //get at least one iteration in
            func(&okays, &dummy); iters++; 

            //execute until end time
            while(true) {
                int64_t now = clock_ns();
                int64_t diff = (int64_t) (clock_ns() - end_time);
                if(diff >= 0)
                    break;

                //perform the operation in "cycles" to amortize
                // against the clock_ns() overhead
                uint64_t next_cycle = iters + cycle;
                for(; iters < next_cycle; iters++)
                    func(&okays, &dummy);
            }
        }
        uint64_t after = clock_ns();

        result->iters = iters;
        result->okays = okays;
        result->duration_ns = after - before;
        control->dummy_sum.fetch_add(dummy);

        notify_stopped();
        while(control->state.load() < state + 1)      
            pause(10);  
    }
}

template<typename Fn1, typename Fn2>
std::vector<std::vector<Bench_Result>> lunch_bench_threads(double seconds, size_t trial_count, Fn1 const& func1, size_t count1, Fn2 const& func2, size_t count2, double cooldown = 0)
{
    size_t count = count1 + count2;
    _Bench_Thread_Control control = {0};
    control.cycle = 1000;
    control.threads = count;

    std::vector<Bench_Result> results(count);
    for(size_t i = 0; i < count1; i++) {
        Bench_Result* res = &results[i];
        std::thread t([&, res]{_bench_thread_func<Fn1>(&control, res, func1);});
        t.detach();
    }
    
    for(size_t i = 0; i < count2; i++) {
        Bench_Result* res = &results[i + count1];
        std::thread t([&, res]{_bench_thread_func<Fn2>(&control, res, func2);});
        t.detach();
    }
        
    uint64_t us = 1000;
    uint64_t ms = 1000'000;
    
    //we repeat (trial) the function multiple times for efficiency.
    //The big part of the expense part of this function is the thread creation. 
    // By repeating just this loop we amortize against it...
    std::vector<std::vector<Bench_Result>> column_results;
    for(size_t trial_i = 1; trial_i <= trial_count; trial_i++) {
        //wait for threads to start
        {
            std::unique_lock<std::mutex> lock(control.waiting_mutex);
            control.waiting.wait(lock, [&]{ return control.begin == trial_i*count; });
        }
    
        //set benchmark start
        uint64_t now = clock_ns();
        control.start_time = now + 16*ms;
        control.end_time = control.start_time + (uint64_t) (seconds/trial_count*1e9) ;
        control.state += 1; 

        //wait for threads to stop
        {
            std::unique_lock<std::mutex> lock(control.waiting_mutex);
            control.waiting.wait(lock, [&]{ return control.end == trial_i*count; });
        }

        //set benchmark end
        control.state += 1; 
        column_results.push_back(results);

        if(cooldown > 0)
            std::this_thread::sleep_for(std::chrono::microseconds((long long) (1e6*cooldown)));
    }

    //end all trials and wait for threads to leave
    control.state = (uint64_t) -1;
    {
        std::unique_lock<std::mutex> lock(control.waiting_mutex);
        control.waiting.wait(lock, [&]{ return control.end == (trial_count + 1)*count; });
    }

    //transpose to be more friendly towards further processing (or outputting)
    std::vector<std::vector<Bench_Result>> all_results(count, std::vector<Bench_Result>(trial_count));
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

struct Escape_Str {
    std::string_view view;
};

std::ostream& operator <<(std::ostream& stream, Escape_Str const& esc) {
    if(esc.view.size() > 0)
        stream << "\"" << esc.view << "\"";
    return stream;
}

void bench_raw_result_to_csv_comment(std::ostream& stream)
{
    stream << Escape_Str{"Version:1. Each line [benchmark_name, operation_name, thread_index, number_of_threads, number_of_trials] then [iters, successful_iters, duration_in_ns] repeating number_of_trials times."} << "\n";
}

void bench_raw_result_to_csv(std::ostream& stream, const char* benchmark_name, size_t thread_count, const std::vector<Bench_Result> results[], const std::string thread_names[], const size_t thread_indices[], size_t results_count)
{
    size_t trials = results_count ? results[0].size() : 0;
    for(size_t i = 0; i < results_count; i++) {
        stream << Escape_Str{benchmark_name} << ", " << Escape_Str{thread_names[i]} << ", " << thread_indices[i] << ", " << thread_count << ", " << trials;
        for(size_t k = 0; k < trials; k++) {
            Bench_Result const& result = results[i][k];
            stream << ", " << result.iters << ", " << result.okays << ", " << result.duration_ns;
        }
        stream << "\n";
    }
}

template<typename Fn1, typename Fn2>
bool bench_process_and_output_csv(double time, size_t trials, size_t min_threads, size_t max_threads, size_t thread_incr, 
    const char* folder, const char* filename, const char* name, 
    Fn1 const& func1, Fn2 const& func2, bool antisym = false, const char* func1_name = "", const char* func2_name = "") {
    
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

    std::vector<std::string> thread_names;
    std::vector<size_t> thread_indices;
    for(size_t i = 0; i < num_runs; i++) {
        thread_indices.push_back(i);
        if(antisym == false || i == 0) 
            thread_names.push_back(func1_name);
        else
            thread_names.push_back(func2_name);
    }

    for(size_t i = min_threads; ; i += thread_incr) {
        if(i >= max_threads)
           i = max_threads; 
     
        num_runs += 1;
        
        printf("threads=%i\n", (int) i);
        std::vector<std::vector<Bench_Result>> raw;
        if(antisym == false)
            raw = lunch_bench_threads<Fn1, decltype(dummy_bench_func)>(time_per_thread, (size_t) trials, func1, (size_t) i, dummy_bench_func, 0);
        else
            raw = lunch_bench_threads<Fn1, Fn2>(time_per_thread, (size_t) trials, func1, 1, func2, i - 1);
        
        bench_raw_result_to_csv(file_raw, name, i, raw.data(), thread_names.data(), thread_indices.data(), i);

        if(i >= max_threads)
            break;
    }

    return file_raw.is_open();
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
void bench_all() {
    uint32_t non_atomic_target = 0;
    std::atomic<uint32_t> target = 0;
    std::atomic_flag flag;
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
    
    auto exchange = [&](uint64_t* okay, uint64_t* dummy){
        *dummy += target.exchange(0, std::memory_order_relaxed);
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
        uint32_t expected = 0;
        target.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        *okay += 1;
    };
    
    auto load_cas = [&](uint64_t* okay, uint64_t* dummy){
        uint32_t expected = target.load(std::memory_order_relaxed);
        *okay += target.compare_exchange_strong(expected, expected + 1, std::memory_order_relaxed);
    };
    
    auto bts = [&](uint64_t* okay, uint64_t* dummy){
        *dummy += atomic_bit_test_and_set(&target, 0);
        *okay += 1;
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

    auto cas_delay0 = [&](uint64_t* okay, uint64_t* dummy){
        uint32_t expected = 0;
        target.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        pause(0);
        *okay += 1;
    };

    auto cas_delay1 = [&](uint64_t* okay, uint64_t* dummy){
        uint32_t expected = 0;
        target.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        pause(1);
        *okay += 1;
    };
    
    auto cas_delay2 = [&](uint64_t* okay, uint64_t* dummy){
        uint32_t expected = 0;
        target.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        pause(2);
        *okay += 1;
    };
    
    auto cas_delay4 = [&](uint64_t* okay, uint64_t* dummy){
        uint32_t expected = 0;
        target.compare_exchange_strong(expected, 1, std::memory_order_relaxed);
        pause(4);
        *okay += 1;
    };

    double time = 4;
    size_t trials = 10;
    size_t min_threads = 1;
    size_t max_threads = (size_t) std::thread::hardware_concurrency();
    size_t incr_threads = 1;
    const char* folder = "bench_data";
    const char* file_atomic = "bench_atomic.csv";
    const char* file_delay = "bench_delay.csv";
    const char* file_pressure = "bench_pressure.csv";

    std::filesystem::remove(std::filesystem::path(folder) / file_atomic);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "load", load, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "store", store, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "exchange", exchange, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "faa", faa, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "and", _and, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "bts", bts, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "cas_all", cas, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "cas_success", load_cas, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "lock", lock, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "read_lock", read_lock, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_atomic, "write_lock", write_lock, dummy_bench_func);
    
    std::filesystem::remove(std::filesystem::path(folder) / file_delay);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_delay, "cas_delay0", cas_delay0, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_delay, "cas_delay1", cas_delay1, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_delay, "cas_delay2", cas_delay2, dummy_bench_func);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_delay, "cas_delay4", cas_delay4, dummy_bench_func);

    std::filesystem::remove(std::filesystem::path(folder) / file_pressure);
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "store_loadp", store, load, true, "op", "load");
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "exchange_loadp", exchange, load, true, "op", "load");
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "faa_loadp", faa, load, true, "op", "load");
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "and_loadp", _and, load, true, "op", "load");
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "bts_loadp", bts, load, true, "op", "load");
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "cas_loadp", cas, load, true, "op", "load");
    bench_process_and_output_csv(time, trials, min_threads, max_threads, incr_threads, folder, file_pressure, "write_lock_loadp", write_lock, read_lock, true, "op", "load");
}