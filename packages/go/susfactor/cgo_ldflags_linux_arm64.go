//go:build linux && arm64

package susfactor

// #cgo LDFLAGS: -L${SRCDIR}/../lib/linux_arm64
import "C"
