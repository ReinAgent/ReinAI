import multiprocessing
import time


def fn(test_round):
    res_list = []
    # handle with res_list

    return (res_list,)


def benchmark(fn, in_file, test_round=5000, conn_num=20):
    keys = []

    thread_pool = multiprocessing.Pool(processes=conn_num)
    results = []
    start_time = time.time()
    for _ in range(conn_num):
        results.append(thread_pool.apply_async(fn, args=(test_round)))
    thread_pool.close()
    thread_pool.join()

    req_all = []
    for item in results:
        req_all += item.get()[0]
    use_time = time.time() - start_time
    print("QPS: " + str(int(conn_num * test_round * 1.0 / use_time)))