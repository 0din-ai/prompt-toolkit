package susfactor

import "fmt"

// SusFactorError is the domain error type for classifier failures.
type SusFactorError struct {
	msg string
}

func (e *SusFactorError) Error() string { return e.msg }

func newError(format string, args ...any) *SusFactorError {
	return &SusFactorError{msg: fmt.Sprintf(format, args...)}
}
