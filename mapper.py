############################################################
# This code creates the mapper graphs shown in the paper
############################################################
import kmapper as km
import distance as d
import numpy as np
import sklearn, csv, json, copy


def read_data(filename):
    """ This function is data processing that is highly unqiue
        to the specific dataset, and unlikely to be of
        interest. """
    with open(filename, "r") as file:
        file_data = csv.reader(file)
        row_length = len(next(file_data))
        file_length = sum(1 for row in file_data)
        #data = np.empty((file_length, row_length - 2))
        data = np.empty((file_length, row_length - 3))

        vars = {}

        with open("D:/fanss/try/code/code_to_readable7.json", "r") as file1:
            codes = set(json.load(file1).keys())

            codes_ordered = [
                             "HC7A",
                             "HC7B",
                             "HC7C",
                             "HC7D",
                             "HC7E",
                             "HC7F",
                             "HC7G",
                             "HC7H",
                             "HC7I",
                            #  "HC7J",
                            #  "HC7K",
                            #  "HC7L",
                            #  "HC7M",
                            #  "HC7N",
                            #  "HC7O",
                             "HC8",
                             "HC9A",
                             "HC9B","HC9C","HC9D",
                             "HC9E",
                             "HC9F",
                             "HC9G",
                             "HC9H",
                            #  "HC9I",
                            #  "HC9J",
                            #  "HC9K",
                            #  "HC9L",
                            #  "HC9M",
                            #  "HC9N",
                            #  "HC9O",
                            #  "HC9P",
                            # "HC9Q",
                             "HC10A",
                             "HC10B",
                             "HC10C",
                             "HC10D",
                             "HC10E",
                             "HC10F",
                             "HC10G",
                            #  "HC10H",
                            #  "HC10I",
                            #  "HC10J",
                            #  "HC10K",
                            #  "HC10L",
                            #  "HC10M",
                            #  "HC10N",
                            #  "HC10O",
                            #  "HC10P",
                            #  "HC10Q",
                            #  "HC10R",
                             "HC11","HC12","HC13","HC14",
                             "HC15","HC17","HC19"
                             ]
           
            # codes_ordered = ["HC8A","HC8B","HC8C","HC8D",
            #                  "HC8E","HC8F","HC8G","HC8H",
            #                  "HC8I","HC8J","HC8K","HC8L",
            #                  "HC8M","HC8N","HC8O","HC8P",
            #                  "HC8Q","HC8R","HC8S","HC8T",
            #                  "HC8U","HC8V","HC9A","HC9B",
            #                  "HC9C","HC9D","HC9E","HC9I",
            #                  "HC9J","HC9K","HC10","HC11",
            #                  "HC13","HC15"]
            
        
            for code in codes:
                vars[code] = np.empty(file_length)

        file.seek(0)
        file_data = csv.DictReader(file)
        count = 0
        for row in file_data:
            for code in codes:
                if code in codes_ordered:
                    data[count][codes_ordered.index(code)] = row[code]
                vars[code][count] = row[code]
            count += 1
    return data, vars


if __name__ == "__main__":

    data, vars = read_data("D:/fanss/try/data.csv")

    mapper = km.KeplerMapper(verbose=1) # instantiate mapper

    # Makes the lens.I used the sum projection
    lens = mapper.fit_transform(data,
                                 projection="sum",
                                 scaler=None)
    
    e = 0.000005
    c = 10 # number of open sets to use in the cover
    o = 30 # overlap  (控制覆盖的大小和重叠程度?)

    # run the algorithm to generate the nodes   #原本作者用的是distance2
    nodes = mapper.map(lens, data,
                       clusterer=sklearn.cluster.DBSCAN(
                           eps=e,
                           min_samples=10,
                           metric=d.independent_distance
                        ),
                        cover=km.Cover(c, o/100))

    # visualize the nodes (writes html files that you can
    # open in a web browser to see the graphs). Each graph's
    # nodes will be colored based on the variable in the file
    # name
    
    for code in vars.keys():
        if not (code == "id"):
            mapper.visualize(nodes,
                title="test",
                color_function_name=code,
                color_function=vars[code],
                custom_tooltips=vars['id'],

                path_html="D:/fanss/try/mappers/" + code + "-"
                              + str(c) + "-" + str(e)
                              + "-" + str(o) + ".html")
