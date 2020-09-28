magnitude = []
ra = []
dec = []
def parse_catalogue(filename):
    star_catalogue = []
    for line in open(filename, 'r'):
        record = line.split()
        magnitude.append(float(record[1]))
        star_catalogue.append({
            'magnitude': float(record[1]),
            'ra': float(record[2]),
            'dec': float(record[3])
        })
    return star_catalogue

if __name__ == "__main__":
    star_catalogue = parse_catalogue('sao60')
    print(len(star_catalogue))
    # print(min(ra), max(ra))
    # print(min(dec), max(dec))
    print(min(magnitude), max(magnitude))
