import re
from urllib.parse import urlparse

class URLFeatures:
    def __init__(self):
        pass

    def get_features(self, url):
        """
        Extracts basic numeric features from a URL string.
        Returns a dictionary of 7 numerical values.
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            path = parsed.path

            features = {
                "url_length": len(url),
                "domain_length": len(domain),
                "num_digits": len(re.findall(r'\d', url)),
                "num_special_chars": len(re.findall(r'[^A-Za-z0-9]', url)),
                "has_https": 1 if parsed.scheme == "https" else 0,
                "num_subdirs": path.count("/"),
                "has_login_keyword": 1 if re.search(r"login|secure|account|update", url, re.IGNORECASE) else 0,
            }
            return features
        except Exception as e:
            print(f"Error extracting features from {url}: {e}")
            return {
                "url_length": 0,
                "domain_length": 0,
                "num_digits": 0,
                "num_special_chars": 0,
                "has_https": 0,
                "num_subdirs": 0,
                "has_login_keyword": 0,
            }
