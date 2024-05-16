import pandas as pd
import numpy as np
# Importing nltk library to do step-1 Tokenization
import nltk as nlt
# Downloading the necessary tokenizer
nlt.download("punkt")
# Reading the data with pandas library
data = pd.read_csv("comcast_consumeraffairs_complaints.csv")

# Importing the word_tokenize
from nltk.tokenize import word_tokenize


# To tokenize defining a function and make all words lower case
def tokenize(text):
    # Checing if text is a string
    if isinstance(text, str):
        tokens = word_tokenize(text)
        return tokens
    # For NaN values or non-string values returning emppty list
    else:
        return []

# Creating a new column named tokens and applying the tokenize function to data["text"]
data["tokens"] = data["text"].apply(tokenize)
# For punctuation
import string
# Stopwords are English words so we use english alphabet
# Used github for this list
stopwords_ = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above",
              "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj",
              "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah",
              "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also",
              "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another",
              "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap",
              "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren",
              "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au",
              "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back",
              "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand",
              "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides",
              "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom",
              "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call",
              "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly",
              "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come",
              "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing",
              "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry",
              "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de",
              "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't",
              "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down",
              "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each",
              "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven",
              "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er",
              "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody",
              "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa",
              "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five",
              "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth",
              "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy",
              "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl",
              "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had",
              "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't",
              "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby",
              "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him",
              "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr",
              "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic",
              "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate",
              "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate",
              "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention",
              "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's",
              "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep",
              "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last",
              "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets",
              "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look",
              "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes",
              "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn",
              "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly",
              "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n",
              "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary",
              "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni",
              "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor",
              "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o",
              "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj",
              "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq",
              "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out",
              "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page",
              "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per",
              "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly",
              "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably",
              "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py",
              "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re",
              "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless",
              "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted",
              "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt",
              "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd",
              "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems",
              "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall",
              "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't",
              "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly",
              "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so",
              "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat",
              "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr",
              "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently",
              "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb",
              "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that",
              "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence",
              "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere",
              "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll",
              "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly",
              "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru",
              "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward",
              "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv",
              "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under",
              "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us",
              "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value",
              "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs",
              "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd",
              "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've",
              "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where",
              "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether",
              "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever",
              "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within",
              "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt",
              "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv",
              "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're",
              "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz"]


def stopwords(text):
    # Checking if text is a list
    if isinstance(text, list):
        # Removing stopwords and punctuation
        removed = [word for word in text if word.lower() not in stopwords_ and word not in string.punctuation]
        return removed
    else:
        return []


data["tokens"] = data["tokens"].apply(stopwords)
def shortening(text):
    cleaned = []
    for word in text:
        if "'" not in word:
            cleaned.append(word)
    cleaned_text = " ".join(cleaned)
    return cleaned_text


# There are still some punctuation and numbers i should remove them
import re


def remove(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Removing spaces
    return cleaned_text.strip().lower()


data["tokens"] = data["tokens"].apply(shortening)

data["tokens"] = data["tokens"].apply(remove)

# To stemming and lemmatization operation
# For stemming by using lemmatization
import spacy as sp

stemmer = sp.load("en_core_web_sm")

# Defining two functions to stemming and lemmatization words
def stem(text):
    doc = stemmer(text)
    stemmed = [word.lemma_ for word in doc]
    return " ".join(stemmed)

# Applying the stem function to data["tokens"]
# It takes some time
data["stemmed"] = data["tokens"].apply(stem)

# Creating a BoW matrix
word_counts = {}
for row in data["stemmed"]:
    for word in row.split():
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

# Words to a list
all_words = list(word_counts.keys())

# BoW matris
Bow_matrix = np.zeros((len(data), len(all_words)))

for i, row in enumerate(data["stemmed"]):
    for word in row.split():
        if word in all_words:
            j = all_words.index(word)
            Bow_matrix[i, j] += 1

# Applying SVD
U, Sigma, V_T = np.linalg.svd(Bow_matrix)

SE
def calculate_MSE(A, A_hat):
    return np.mean((A - A_hat) ** 2)


def calculate_FN(A, A_hat):
    return np.linalg.norm(A - A_hat)


t, d = Bow_matrix.shape
k_range = range(10, min(t, d) // 10 + 1, 20)

best_k_MSE = None
best_MSE = float('inf')

best_k_FN = None
best_FN = float('inf')

for k in k_range:
    U_k = U[:, :k]
    Sigma_k = np.diag(Sigma[:k])
    V_T_k = V_T[:k, :]

    A_hat = np.dot(np.dot(U_k, Sigma_k), V_T_k)

    MSE = calculate_MSE(Bow_matrix, A_hat)
    FN = calculate_FN(Bow_matrix, A_hat)

    if MSE < best_MSE:
        best_k_MSE = k
        best_MSE = MSE

    if FN < best_FN:
        best_k_FN = k
        best_FN = FN

print("Optimum k for MSE:", best_k_MSE)
print("MSE:", best_MSE)
print("Optimum k for FN:", best_k_FN)
print("FN:", best_FN)

#Query-document cosine similarity calculation using Equation.10
def calculate_cosine_similarity(query_vector, document_matrix):
    similarities = []
    for doc_vector in document_matrix:
        dot_product = np.dot(query_vector, doc_vector)
        norm_query = np.linalg.norm(query_vector)
        norm_doc = np.linalg.norm(doc_vector)

        if norm_query != 0 and norm_doc != 0:
            similarity = dot_product / (norm_query * norm_doc)
        else:
            similarity = 0

        similarities.append(similarity)

    return similarities


# Define the queries
queries = dict(query1=['ignorant', 'overwhelming'], query2=['xfinity', 'frustrate', 'adapter', 'verizon', 'router'],
               query3=['terminate', 'rent', 'promotion', 'joke', 'liar', 'internet', 'horrible'],
               query4=['kindergarten', 'ridiculous', 'internet', 'clerk', 'terrible'])


def create_query_vector(query_terms, all_terms):
    # Initialize query vector with zeros
    query_vector = np.zeros(len(all_terms))

    # Iterate over each term in the query
    for term in query_terms:
        # Check if the term exists in the list of all terms
        if term in all_terms:
            # Get the index of the term in the list of all terms
            index = all_terms.index(term)
            # Set the value at that index to 1
            query_vector[index] = 1

    return query_vector


#Calculate the query vectors
query_vectors = {}
for query_name, query_words in queries.items():
    query_vectors[query_name] = create_query_vector(query_words, all_words)

#Find the most relevant document for each query
for query_name, query_vector in query_vectors.items():
    similarities = calculate_cosine_similarity(query_vector, Bow_matrix)
    most_relevant_index = np.argmax(similarities)
    similarity = similarities[most_relevant_index]
    print(f"Most relevant document for {query_name}: Document {most_relevant_index + 1}, Similarity: {similarity:.2f}")