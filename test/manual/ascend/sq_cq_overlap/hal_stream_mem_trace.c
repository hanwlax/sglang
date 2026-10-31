/*
 * Trace the real HAL arguments without changing them or locking across calls.
 * Loaded through LD_PRELOAD (PLT/GOT) and glibc LD_AUDIT (explicit dlsym).
 * Both instances avoid wrapping each other; records identify their instance.
 *
 * Public ABI reference: cann/driver pkg_inc/ascend_hal_define.h,
 * struct sq_switch_stream_info (64-bit layout: 24 bytes, stream_mem at 16).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <link.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#ifdef SQ_CQ_USE_DRIVER_HEADERS
#include <ascend_hal_define.h>
#else
struct sq_switch_stream_info {
  uint32_t sq_id;
  uint32_t stream_id;
  uint32_t sq_depth;
  uint32_t rsv;
  void *stream_mem;
};
#endif

_Static_assert(sizeof(void *) == 8, "Only 64-bit HAL is supported");
_Static_assert(sizeof(struct sq_switch_stream_info) == 24,
               "Driver ABI differs: do not trace with this layout");
_Static_assert(offsetof(struct sq_switch_stream_info, sq_id) == 0 &&
                   offsetof(struct sq_switch_stream_info, stream_id) == 4 &&
                   offsetof(struct sq_switch_stream_info, sq_depth) == 8,
               "Driver queue/stream/depth offsets differ");
_Static_assert(offsetof(struct sq_switch_stream_info, stream_mem) == 16,
               "Driver stream_mem offset differs");

typedef int (*switch_fn)(uint32_t, struct sq_switch_stream_info *, uint32_t);
typedef int (*fill_fn)(uint32_t, uint32_t, void *, void *, uint32_t);
static _Atomic(switch_fn) real_switch;
static _Atomic(fill_fn) real_fill;
static _Atomic uint64_t next_call = 1;
static int audit_instance;

static uint64_t now_ns(clockid_t clock_id) {
  struct timespec value;
  if (clock_gettime(clock_id, &value) != 0) {
    return 0;
  }
  return (uint64_t)value.tv_sec * 1000000000ULL + (uint64_t)value.tv_nsec;
}

static void emit(uint64_t mono_ns, const char *format, ...) {
  int saved_errno = errno;
  const char *directory = getenv("SQ_CQ_TRACE_DIR");
  if (directory == NULL) {
    errno = saved_errno;
    return;
  }
  char path[PATH_MAX];
  int count = snprintf(path, sizeof(path), "%s/hal.%ld.jsonl", directory,
                       (long)getpid());
  if (count < 0 || (size_t)count >= sizeof(path)) {
    errno = saved_errno;
    return;
  }
  char line[1024];
  int prefix =
      snprintf(line, sizeof(line),
               "{\"probe\":\"%s\",\"pid\":%ld,\"tid\":%ld,\"mono_ns\":%" PRIu64
               ",\"wall_ns\":%" PRIu64 ",",
               audit_instance ? "audit" : "preload", (long)getpid(),
               (long)syscall(SYS_gettid), mono_ns, now_ns(CLOCK_REALTIME));
  if (prefix <= 0 || (size_t)prefix >= sizeof(line)) {
    errno = saved_errno;
    return;
  }
  va_list args;
  va_start(args, format);
  int body =
      vsnprintf(line + prefix, sizeof(line) - (size_t)prefix, format, args);
  va_end(args);
  if (body < 0 || (size_t)body + (size_t)prefix + 2 >= sizeof(line)) {
    errno = saved_errno;
    return;
  }
  size_t length = (size_t)prefix + (size_t)body;
  line[length++] = '}';
  line[length++] = '\n';
  /* One append per record, no shared userspace lock and no buffered stdio.
   * Open per record also avoids inherited-fd/lock problems after fork. */
  int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
  if (fd >= 0) {
    ssize_t written;
    do {
      written = write(fd, line, length);
    } while (written < 0 && errno == EINTR);
    (void)close(fd);
  }
  errno = saved_errno;
}

static void switch_record(const char *event, uint64_t call, uint64_t stamp,
                          uint32_t device, struct sq_switch_stream_info *info,
                          uint32_t num, int ret) {
  if (info == NULL || num == 0) {
    emit(stamp,
         "\"event\":\"switch_empty\",\"call\":%" PRIu64
         ",\"device\":%u,\"num\":%u,\"ret\":%d",
         call, device, num, ret);
    return;
  }
  for (uint32_t i = 0; i < num; i++) {
    emit(stamp,
         "\"event\":\"%s\",\"call\":%" PRIu64
         ",\"index\":%u,\"device\":%u,\"stream\":%u,\"sq\":%u,"
         "\"depth\":%u,\"stream_mem\":\"0x%" PRIxPTR "\",\"num\":%u,\"ret\":%d",
         event, call, i, device, info[i].stream_id, info[i].sq_id,
         info[i].sq_depth, (uintptr_t)info[i].stream_mem, num, ret);
  }
}

static int trace_switch(uint32_t device, struct sq_switch_stream_info *info,
                        uint32_t num) {
  uint64_t call = atomic_fetch_add(&next_call, 1);
  switch_record("switch_begin", call, now_ns(CLOCK_MONOTONIC), device, info,
                num, 0);
  int ret = atomic_load(&real_switch)(device, info, num);
  int saved_errno = errno;
  switch_record("switch_end", call, now_ns(CLOCK_MONOTONIC), device, info, num,
                ret);
  errno = saved_errno;
  return ret;
}

static int trace_fill(uint32_t device, uint32_t stream, void *stream_mem,
                      void *task_info, uint32_t task_cnt) {
  uint64_t call = atomic_fetch_add(&next_call, 1);
  emit(now_ns(CLOCK_MONOTONIC),
       "\"event\":\"fill_begin\",\"call\":%" PRIu64
       ",\"device\":%u,\"stream\":%u,\"stream_mem\":\"0x%" PRIxPTR
       "\",\"task_info\":\"0x%" PRIxPTR "\",\"task_cnt\":%u",
       call, device, stream, (uintptr_t)stream_mem, (uintptr_t)task_info,
       task_cnt);
  int ret =
      atomic_load(&real_fill)(device, stream, stream_mem, task_info, task_cnt);
  int saved_errno = errno;
  emit(now_ns(CLOCK_MONOTONIC),
       "\"event\":\"fill_end\",\"call\":%" PRIu64
       ",\"device\":%u,\"stream\":%u,\"stream_mem\":\"0x%" PRIxPTR
       "\",\"task_cnt\":%u,\"ret\":%d",
       call, device, stream, (uintptr_t)stream_mem, task_cnt, ret);
  errno = saved_errno;
  return ret;
}

/* Normal ELF interposition covers non-PLT/GOT calls too. Do not interpose
 * dlsym itself: that would change RTLD_NEXT lookup semantics for other code. */
int halSqSwitchStreamBatch(uint32_t device, struct sq_switch_stream_info *info,
                           uint32_t num) {
  if (atomic_load(&real_switch) == NULL) {
    switch_fn function = (switch_fn)dlsym(RTLD_NEXT, "halSqSwitchStreamBatch");
    if (function == NULL) {
      static const char error[] =
          "SQ/CQ trace: cannot resolve real halSqSwitchStreamBatch\n";
      ssize_t ignored = write(STDERR_FILENO, error, sizeof(error) - 1);
      (void)ignored;
      _exit(127);
    }
    atomic_store(&real_switch, function);
  }
  return trace_switch(device, info, num);
}

int halStreamTaskFill(uint32_t device, uint32_t stream, void *stream_mem,
                      void *task_info, uint32_t task_cnt) {
  if (atomic_load(&real_fill) == NULL) {
    fill_fn function = (fill_fn)dlsym(RTLD_NEXT, "halStreamTaskFill");
    if (function == NULL) {
      static const char error[] =
          "SQ/CQ trace: cannot resolve real halStreamTaskFill\n";
      ssize_t ignored = write(STDERR_FILENO, error, sizeof(error) - 1);
      (void)ignored;
      _exit(127);
    }
    atomic_store(&real_fill, function);
  }
  return trace_fill(device, stream, stream_mem, task_info, task_cnt);
}

unsigned int la_version(unsigned int version) {
  if (version < LAV_CURRENT) {
    return 0;
  }
  audit_instance = 1;
  emit(now_ns(CLOCK_MONOTONIC),
       "\"event\":\"probe_loaded\",\"abi_size\":%zu,"
       "\"stream_mem_offset\":%zu",
       sizeof(struct sq_switch_stream_info),
       offsetof(struct sq_switch_stream_info, stream_mem));
  return LAV_CURRENT;
}

unsigned int la_objopen(struct link_map *map, Lmid_t lmid, uintptr_t *cookie) {
  (void)lmid;
  *cookie = (uintptr_t)map;
  return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
                       uintptr_t *defcook, unsigned int *flags,
                       const char *name) {
  (void)ndx;
  *flags |= LA_SYMB_NOPLTENTER | LA_SYMB_NOPLTEXIT;
  struct link_map *reference = (struct link_map *)*refcook;
  struct link_map *definition = (struct link_map *)*defcook;
  const char *ref_name = strrchr(reference->l_name, '/');
  const char *def_name = strrchr(definition->l_name, '/');
  ref_name = ref_name ? ref_name + 1 : reference->l_name;
  def_name = def_name ? def_name + 1 : definition->l_name;
  /* Avoid wrapping our preload wrapper or its RTLD_NEXT lookup again. */
  if (strcmp(ref_name, "libsqcq_hal_trace.so") == 0 ||
      strcmp(def_name, "libsqcq_hal_trace.so") == 0) {
    return sym->st_value;
  }
  if (strcmp(name, "halSqSwitchStreamBatch") == 0) {
    switch_fn previous = atomic_load(&real_switch);
    if (previous != NULL && previous != (switch_fn)sym->st_value) {
      emit(now_ns(CLOCK_MONOTONIC), "\"event\":\"unsupported_provider\","
                                    "\"api\":\"halSqSwitchStreamBatch\"");
      return sym->st_value;
    }
    atomic_store(&real_switch, (switch_fn)sym->st_value);
    emit(now_ns(CLOCK_MONOTONIC),
         "\"event\":\"symbol_bound\","
         "\"api\":\"halSqSwitchStreamBatch\",\"dlsym\":%s",
         (*flags & LA_SYMB_DLSYM) ? "true" : "false");
    return (uintptr_t)trace_switch;
  }
  if (strcmp(name, "halStreamTaskFill") == 0) {
    fill_fn previous = atomic_load(&real_fill);
    if (previous != NULL && previous != (fill_fn)sym->st_value) {
      emit(now_ns(CLOCK_MONOTONIC), "\"event\":\"unsupported_provider\","
                                    "\"api\":\"halStreamTaskFill\"");
      return sym->st_value;
    }
    atomic_store(&real_fill, (fill_fn)sym->st_value);
    emit(now_ns(CLOCK_MONOTONIC),
         "\"event\":\"symbol_bound\","
         "\"api\":\"halStreamTaskFill\",\"dlsym\":%s",
         (*flags & LA_SYMB_DLSYM) ? "true" : "false");
    return (uintptr_t)trace_fill;
  }
  return sym->st_value;
}
