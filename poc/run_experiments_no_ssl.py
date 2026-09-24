"""Run experiments with SSL verification disabled (for corporate proxy environments)."""
import ssl
import os
import sys
import warnings

# Disable SSL verification before any imports
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'

warnings.filterwarnings('ignore')

# Aggressively patch requests to disable SSL at the lowest level
import requests
from requests.adapters import HTTPAdapter

class NoSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        kwargs['cert_reqs'] = ssl.CERT_NONE
        kwargs['assert_hostname'] = False
        return super().init_poolmanager(*args, **kwargs)

# Monkey-patch requests.Session to always use our adapter
_original_session_init = requests.Session.__init__
def patched_session_init(self, *args, **kwargs):
    _original_session_init(self, *args, **kwargs)
    self.mount('https://', NoSSLAdapter())
    self.mount('http://', HTTPAdapter())
    self.verify = False
requests.Session.__init__ = patched_session_init

# Now import and run experiments
if __name__ == "__main__":
    # Import experiments module after SSL is disabled
    import experiments
    experiments.main()
