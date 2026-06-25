//go:build darwin && arm64

package susfactor

// #cgo LDFLAGS: -L${SRCDIR}/../lib/darwin_arm64
import "C"
