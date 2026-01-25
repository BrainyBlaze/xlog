#define _GNU_SOURCE

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// A tiny LD_PRELOAD shim that emulates cross-directory rename() when the
// underlying sandboxed filesystem returns EXDEV for renames across dirs.
//
// Rustc/cargo use "write in temp dir then rename into place" for rmeta/rlib,
// which fails in this environment because rename() across directories always
// returns EXDEV. For build artifacts, copy+atomic-rename-in-dest-dir is fine.

static int copy_file_fd(int src_fd, int dst_fd) {
  char buf[1024 * 1024];
  for (;;) {
    ssize_t n = read(src_fd, buf, sizeof(buf));
    if (n == 0) {
      return 0;
    }
    if (n < 0) {
      return -1;
    }
    ssize_t off = 0;
    while (off < n) {
      ssize_t w = write(dst_fd, buf + off, (size_t)(n - off));
      if (w < 0) {
        return -1;
      }
      off += w;
    }
  }
}

static int exdev_copy_then_replace(const char *oldpath, const char *newpath) {
  struct stat st;
  if (stat(oldpath, &st) != 0) {
    return -1;
  }
  if (!S_ISREG(st.st_mode)) {
    errno = EXDEV;
    return -1;
  }

  int src_fd = open(oldpath, O_RDONLY | O_CLOEXEC);
  if (src_fd < 0) {
    return -1;
  }

  // Create a temp file in the destination directory, then rename within that
  // directory (allowed in this sandbox). We keep the temp name hidden.
  char tmp[PATH_MAX];
  const char *slash = strrchr(newpath, '/');
  if (slash) {
    size_t dir_len = (size_t)(slash - newpath);
    if (dir_len + 1 + strlen(".exdevtmpXXXXXX") + 1 > sizeof(tmp)) {
      close(src_fd);
      errno = ENAMETOOLONG;
      return -1;
    }
    memcpy(tmp, newpath, dir_len);
    tmp[dir_len] = '\0';
    strncat(tmp, "/.exdevtmpXXXXXX", sizeof(tmp) - strlen(tmp) - 1);
  } else {
    strncpy(tmp, ".exdevtmpXXXXXX", sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
  }

  int dst_fd = mkstemp(tmp);
  if (dst_fd < 0) {
    close(src_fd);
    return -1;
  }

  // Best-effort: keep permissions similar.
  (void)fchmod(dst_fd, st.st_mode & 0777);

  if (copy_file_fd(src_fd, dst_fd) != 0) {
    int saved = errno;
    close(src_fd);
    close(dst_fd);
    unlink(tmp);
    errno = saved;
    return -1;
  }

  (void)fsync(dst_fd);
  close(src_fd);
  close(dst_fd);

  // Now replace the destination (rename within the same directory).
  int (*real_rename)(const char *, const char *) =
      (int (*)(const char *, const char *))dlsym(RTLD_NEXT, "rename");
  if (!real_rename) {
    // Shouldn't happen; fall back to the raw syscall symbol via glibc.
    unlink(tmp);
    errno = ENOSYS;
    return -1;
  }
  if (real_rename(tmp, newpath) != 0) {
    int saved = errno;
    unlink(tmp);
    errno = saved;
    return -1;
  }

  // Finally remove the source path.
  (void)unlink(oldpath);
  return 0;
}

int rename(const char *oldpath, const char *newpath) {
  int (*real_rename)(const char *, const char *) =
      (int (*)(const char *, const char *))dlsym(RTLD_NEXT, "rename");
  if (!real_rename) {
    errno = ENOSYS;
    return -1;
  }

  int rc = real_rename(oldpath, newpath);
  if (rc == 0) {
    return 0;
  }
  if (errno != EXDEV) {
    return rc;
  }

  // Fallback for this sandbox.
  return exdev_copy_then_replace(oldpath, newpath);
}
