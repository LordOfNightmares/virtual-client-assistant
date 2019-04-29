question = "How much is software? What is the price of the software? What is the cost of software?"

# import urllib.request
#
import nltk
# from bs4 import BeautifulSoup
from nltk.corpus import stopwords
# nltk.download('punkt')
# nltk.download('stopwords')
# response = urllib.request.urlopen('https://www.zap.md/laptop-pc/monitoare')
# html = response.read()
# soup = BeautifulSoup(html, "html5lib")
# text = soup.get_text(strip=True)
tokens = [t for t in text.split()]
# clean_tokens = question[:]
# sr = stopwords.words('english')
nltk.sent_tokenize(question)

# for token in :
#     if token in stopwords.words('english'):
#         clean_tokens.remove(token)
freq = nltk.FreqDist(clean_tokens)
for key, val in freq.items():
    print(str(key) + ':' + str(val))
freq.plot(30, cumulative=False)
