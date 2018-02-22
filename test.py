from nltk.translate.bleu_score import sentence_bleu,SmoothingFunction


def bleu(reference,candidate):
    cc = SmoothingFunction()
    score = sentence_bleu(reference, candidate)
    return score

ref = ['not','being','awesome']
cand = ['not','being','awesome']
print(bleu(ref,cand))