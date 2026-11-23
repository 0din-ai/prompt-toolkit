package susfactor

import (
	"errors"
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

type fakeIntraOpSetter struct {
	calls []int
	err   error
}

func (f *fakeIntraOpSetter) SetIntraOpNumThreads(n int) error {
	f.calls = append(f.calls, n)
	return f.err
}

func TestWithIntraOpThreads(t *testing.T) {
	t.Run("absent option leaves zero value", func(t *testing.T) {
		cfg := &config{}
		if cfg.intraOpThreads != 0 {
			t.Fatalf("default intraOpThreads=%d, want 0", cfg.intraOpThreads)
		}
	})
	t.Run("option stores n", func(t *testing.T) {
		cfg := &config{}
		WithIntraOpThreads(3)(cfg)
		if cfg.intraOpThreads != 3 {
			t.Fatalf("intraOpThreads=%d, want 3", cfg.intraOpThreads)
		}
	})
}

func TestApplyIntraOpThreads(t *testing.T) {
	for _, n := range []int{0, -1, -100} {
		f := &fakeIntraOpSetter{}
		if err := applyIntraOpThreads(f, n); err != nil {
			t.Errorf("n=%d: unexpected error %v", n, err)
		}
		if len(f.calls) != 0 {
			t.Errorf("n=%d: SetIntraOpNumThreads called %v, want no call", n, f.calls)
		}
	}

	t.Run("positive n is forwarded once", func(t *testing.T) {
		f := &fakeIntraOpSetter{}
		if err := applyIntraOpThreads(f, 2); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(f.calls) != 1 || f.calls[0] != 2 {
			t.Fatalf("calls=%v, want [2]", f.calls)
		}
	})

	t.Run("setter error is returned", func(t *testing.T) {
		want := errors.New("boom")
		if err := applyIntraOpThreads(&fakeIntraOpSetter{err: want}, 2); !errors.Is(err, want) {
			t.Fatalf("err=%v, want %v", err, want)
		}
	})
}

// TestApplyIntraOpThreads_realORT checks the real ort.SessionOptions accepts the
// value. Skips when ORT_LIB_PATH is not set.
func TestApplyIntraOpThreads_realORT(t *testing.T) {
	libPath := os.Getenv("ORT_LIB_PATH")
	if libPath == "" {
		t.Skip("ORT_LIB_PATH not set — skipping real ORT session options test")
	}
	if err := initORT(libPath); err != nil {
		t.Fatalf("initORT: %v", err)
	}
	so, err := ort.NewSessionOptions()
	if err != nil {
		t.Fatalf("NewSessionOptions: %v", err)
	}
	defer so.Destroy()
	for _, n := range []int{0, -1, 1, 2} {
		if err := applyIntraOpThreads(so, n); err != nil {
			t.Errorf("n=%d: %v", n, err)
		}
	}
}
