import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENAI_MODEL", "gpt-4")

if 'boto3' not in sys.modules:
    sys.modules['boto3'] = types.SimpleNamespace(client=lambda *args, **kwargs: None)

if 'jwt' not in sys.modules:
    jwt_module = types.ModuleType('jwt')
    jwt_module.encode = lambda *args, **kwargs: "token"
    jwt_module.decode = lambda *args, **kwargs: {}
    exceptions_mod = types.ModuleType('jwt.exceptions')
    exceptions_mod.ExpiredSignatureError = Exception
    exceptions_mod.InvalidTokenError = Exception
    jwt_module.exceptions = exceptions_mod
    sys.modules['jwt'] = jwt_module
    sys.modules['jwt.exceptions'] = exceptions_mod

if 'stripe' not in sys.modules:
    stripe_module = types.ModuleType('stripe')
    stripe_module.checkout = types.SimpleNamespace(Session=types.SimpleNamespace(create=lambda **kwargs: None))
    stripe_module.Webhook = types.SimpleNamespace(construct_event=lambda payload, sig, secret: {})
    stripe_module.Subscription = types.SimpleNamespace(modify=lambda *args, **kwargs: None)
    stripe_module.Invoice = types.SimpleNamespace(list=lambda **kwargs: {'data': []})
    stripe_module.error = types.SimpleNamespace(StripeError=Exception, SignatureVerificationError=Exception)
    sys.modules['stripe'] = stripe_module

if 'cv2' not in sys.modules:
    cv2_stub = types.ModuleType('cv2')
    cv2_stub.FONT_HERSHEY_SIMPLEX = 0
    cv2_stub.COLOR_BGR2RGB = 0
    cv2_stub.cvtColor = lambda img, code: img
    sys.modules['cv2'] = cv2_stub
