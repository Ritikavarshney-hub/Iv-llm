"""#!/usr/bin/env python3
Standalone attention fp16 masking test (no pytest needed).

Usage:
    python tests/run_attention_test.py

Exit codes:
    0  success
    1  assertion failure
    2  missing dependency (torch)

import sys
import os

try:
    import torch
except Exception as e:
    print("ERROR: torch is not installed. Install PyTorch to run this test.", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(2)

# allow importing from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from files.model import MultiHeadAttention


def run_test():
    Q = torch.randn(1, 2, 3, 4, dtype=torch.float16, device='cpu')
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    mask = torch.ones(1, 2, 3, 3, dtype=torch.uint8)
    mask[0, 0, :, 1] = 0

    out, p_attn = MultiHeadAttention.attention(Q, K, V, mask, torch.nn.Dropout(0.0))

    assert out.dtype == Q.dtype, f"out.dtype {out.dtype} != Q.dtype {Q.dtype}"
    assert p_attn.dtype == Q.dtype, f"p_attn.dtype {p_attn.dtype} != Q.dtype {Q.dtype}"
    assert out.shape == Q.shape, f"out.shape {out.shape} != Q.shape {Q.shape}"
    assert torch.isfinite(out).all(), "out contains non-finite values"
    assert torch.isfinite(p_attn).all(), "p_attn contains non-finite values"


if __name__ == '__main__':
    try:
        run_test()
    except AssertionError as e:
        print('FAIL:', e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(1)
    else:
        print('PASS: attention fp16 masking test')
        sys.exit(0)
#!/usr/bin/env python3
import sys
import os

# allow importing package modules from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
except Exception:
    print('Skipping test: torch is not installed.')
    sys.exit(0)

from files.model import MultiHeadAttention


def main():
    try:
        Q = torch.randn(1, 2, 3, 4, dtype=torch.float16, device='cpu')
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)

        mask = torch.ones(1, 2, 3, 3, dtype=torch.uint8)
        mask[0, 0, :, 1] = 0

        out, p_attn = MultiHeadAttention.attention(Q, K, V, mask, torch.nn.Dropout(0.0))

        assert out.dtype == Q.dtype
        assert p_attn.dtype == Q.dtype
        assert out.shape == Q.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(p_attn).all()

    except AssertionError as e:
        print('TEST FAILED:', e)
        sys.exit(2)
    except Exception as e:
        print('ERROR during test:', e)
        sys.exit(3)

    print('TEST PASSED')
    sys.exit(0)


if __name__ == '__main__':
    main()
"""
#!/usr/bin/env python3
"""Standalone attention fp16 masking test (no pytest needed).

Usage:
    python tests/run_attention_test.py

Exit codes:
    0  success
    1  failure
    2  missing dependency (torch)
"""
import sys
import os

try:
    import torch
except Exception as e:
    print("ERROR: torch is not installed. Install PyTorch to run this test.", file=sys.stderr)
    print(e, file=sys.stderr)
    sys.exit(2)

# allow importing from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from files.model import MultiHeadAttention


def run_test():
    Q = torch.randn(1, 2, 3, 4, dtype=torch.float16, device='cpu')
    K = torch.randn_like(Q)
    V = torch.randn_like(Q)

    mask = torch.ones(1, 2, 3, 3, dtype=torch.uint8)
    mask[0, 0, :, 1] = 0

    out, p_attn = MultiHeadAttention.attention(Q, K, V, mask, torch.nn.Dropout(0.0))

    assert out.dtype == Q.dtype, f"out.dtype {out.dtype} != Q.dtype {Q.dtype}"
    assert p_attn.dtype == Q.dtype, f"p_attn.dtype {p_attn.dtype} != Q.dtype {Q.dtype}"
    assert out.shape == Q.shape, f"out.shape {out.shape} != Q.shape {Q.shape}"
    assert torch.isfinite(out).all(), "out contains non-finite values"
    assert torch.isfinite(p_attn).all(), "p_attn contains non-finite values"


if __name__ == '__main__':
    try:
        run_test()
    except AssertionError as e:
        print('FAIL:', e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print('ERROR:', e, file=sys.stderr)
        sys.exit(1)
    else:
        print('PASS: attention fp16 masking test')
        sys.exit(0)
