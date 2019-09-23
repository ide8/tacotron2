path = 'metadata_lj.csv'

dist = {}
with open(path, 'r') as f:
    for l in f:
        striped = l.strip().split('|')
        sentence = striped[1]

        c = sentence.count(' ')
        dist[c] = dist.get(c, 0) + 1

print(sorted(dist.items(), key=lambda x: -x[0]))
