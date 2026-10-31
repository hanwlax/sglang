/* Fake HAL + client for --self-test; never loaded by a real server run. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

struct sq_switch_stream_info {
  uint32_t sq_id, stream_id, sq_depth, rsv;
  void *stream_mem;
};

#ifdef SQ_CQ_FAKE_LIBRARY
static uintptr_t cached[2][32];
static _Atomic int concurrent_entered;
static _Atomic int concurrent_release;

int fake_switch_entered(void) { return atomic_load(&concurrent_entered); }

int halSqSwitchStreamBatch(uint32_t device, struct sq_switch_stream_info *info,
                           uint32_t num) {
  for (uint32_t i = 0; i < num; i++) {
    uint32_t stream = info[i].stream_id;
    if (device > 1 || stream >= 32) {
      return 71;
    }
    if (stream == 24) {
      return 7;
    }
    if (stream == 27) {
      atomic_store(&concurrent_entered, 1);
      for (int attempt = 0; !atomic_load(&concurrent_release); attempt++) {
        if (attempt == 2000) {
          return 99;
        }
        usleep(1000);
      }
    }
    /* Model a first-bind-only cached address, like the code under study. */
    if (cached[device][stream] == 0) {
      cached[device][stream] = (uintptr_t)info[i].stream_mem;
    }
    if (stream == 23) {
      info[i].stream_mem = (void *)((uintptr_t)info[i].stream_mem + 64);
    }
  }
  errno = EAGAIN;
  return 0;
}

int halStreamTaskFill(uint32_t device, uint32_t stream, void *memory,
                      void *task_info, uint32_t count) {
  if (task_info != (void *)0xabc0 || count != 127 || device > 1 ||
      stream >= 32) {
    return 72;
  }
  if (stream == 27) {
    atomic_store(&concurrent_release, 1);
    return 0;
  }
  uintptr_t base = cached[device][stream];
  if (base == 0) {
    return 33;
  }
  if ((uintptr_t)memory < base ||
      (uintptr_t)memory + count * 64 > base + 0x7e40) {
    return 14;
  }
  errno = EBUSY;
  return 0;
}
#else
typedef int (*switch_fn)(uint32_t, struct sq_switch_stream_info *, uint32_t);
typedef int (*fill_fn)(uint32_t, uint32_t, void *, void *, uint32_t);
static switch_fn switch_api;
static fill_fn fill_api;
static int (*entered_api)(void);

#ifdef SQ_CQ_FAKE_LINKED
extern int halSqSwitchStreamBatch(uint32_t, struct sq_switch_stream_info *,
                                  uint32_t);
extern int halStreamTaskFill(uint32_t, uint32_t, void *, void *, uint32_t);
extern int fake_switch_entered(void);
static int call_switch(uint32_t d, struct sq_switch_stream_info *i,
                       uint32_t n) {
  return halSqSwitchStreamBatch(d, i, n);
}
static int call_fill(uint32_t d, uint32_t s, void *m, void *t, uint32_t n) {
  return halStreamTaskFill(d, s, m, t, n);
}
#endif

#define CHECK(test)                                                            \
  do {                                                                         \
    if (!(test)) {                                                             \
      fprintf(stderr, "fixture failure at line %d: %s\n", __LINE__, #test);    \
      exit(80);                                                                \
    }                                                                          \
  } while (0)

static struct sq_switch_stream_info binding(uint32_t stream, uintptr_t memory) {
  struct sq_switch_stream_info info = {38, stream, 505, 0, (void *)memory};
  return info;
}

static void *overlapping_switch(void *unused) {
  (void)unused;
  struct sq_switch_stream_info info = binding(27, 0x1208bfe00000ULL);
  CHECK(switch_api(0, &info, 1) == 0);
  return NULL;
}

int main(int argc, char **argv) {
#ifdef SQ_CQ_FAKE_LINKED
  (void)argc;
  (void)argv;
  switch_api = call_switch;
  fill_api = call_fill;
  entered_api = fake_switch_entered;
#else
  CHECK(argc == 2);
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  CHECK(handle != NULL);
  switch_api = (switch_fn)dlsym(handle, "halSqSwitchStreamBatch");
  fill_api = (fill_fn)dlsym(handle, "halStreamTaskFill");
  entered_api = (int (*)(void))dlsym(handle, "fake_switch_entered");
  CHECK(switch_api && fill_api && entered_api);
#endif
  struct sq_switch_stream_info info = binding(22, 0x1208bfe00000ULL);
  CHECK(switch_api(0, &info, 1) == 0);
  CHECK(errno == EAGAIN);
  CHECK(fill_api(0, 22, info.stream_mem, (void *)0xabc0, 127) == 0);
  CHECK(errno == EBUSY);
#ifdef SQ_CQ_FAKE_MIXED
  CHECK(argc == 2);
  void *handle = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
  CHECK(handle != NULL);
  switch_api = (switch_fn)dlsym(handle, "halSqSwitchStreamBatch");
  fill_api = (fill_fn)dlsym(handle, "halStreamTaskFill");
  CHECK(switch_api && fill_api);
#endif
  info.stream_mem = (void *)0x1208bfe08000ULL;
  CHECK(switch_api(0, &info, 1) == 0);
  /* Switch and Fill have identical inputs, but the fake driver kept an old
   * base. */
  CHECK(fill_api(0, 22, info.stream_mem, (void *)0xabc0, 127) == 14);
  /* Fill differs from the latest switch input. */
  CHECK(fill_api(0, 22, (void *)0x1208bfe00000ULL, (void *)0xabc0, 127) == 0);
  info = binding(23, 0x1208bfe00000ULL);
  CHECK(switch_api(0, &info, 1) == 0);
  CHECK(info.stream_mem == (void *)0x1208bfe00040ULL);
  CHECK(fill_api(0, 23, info.stream_mem, (void *)0xabc0, 127) == 0);
  info = binding(24, 0x1208bfe00000ULL);
  CHECK(switch_api(0, &info, 1) == 7);
  CHECK(fill_api(0, 24, info.stream_mem, (void *)0xabc0, 127) == 33);
  struct sq_switch_stream_info batch[] = {
      {40, 25, 505, 0, (void *)0x1208bfe00000ULL},
      {41, 26, 505, 0, (void *)0x1208bfe08000ULL},
  };
  CHECK(switch_api(1, batch, 2) == 0);
  CHECK(fill_api(1, 25, batch[0].stream_mem, (void *)0xabc0, 127) == 0);
  CHECK(fill_api(1, 26, batch[1].stream_mem, (void *)0xabc0, 127) == 0);
  pthread_t thread;
  CHECK(pthread_create(&thread, NULL, overlapping_switch, NULL) == 0);
  for (int attempt = 0; !entered_api(); attempt++) {
    CHECK(attempt < 2000);
    usleep(1000);
  }
  CHECK(fill_api(0, 27, (void *)0x1208bfe00000ULL, (void *)0xabc0, 127) == 0);
  CHECK(pthread_join(thread, NULL) == 0);
  pid_t child = fork();
  CHECK(child >= 0);
  if (child == 0) {
    CHECK(fill_api(0, 22, (void *)0x1208bfe00000ULL, (void *)0xabc0, 127) == 0);
    _exit(0);
  }
  int status;
  CHECK(waitpid(child, &status, 0) == child);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  puts("fake HAL: arguments, return codes, errno, mutation, threads and fork "
       "passed");
  return 0;
}
#endif
