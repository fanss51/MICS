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
        # data = np.empty((file_length, row_length - 3))
        data = np.empty((file_length, row_length))

        vars = {}

        with open("D:/fanss/try/code/code_to_readableS5.json", "r") as file1:
            codes = set(json.load(file1).keys())

            codes_ordered = ["id","HC8A","HC8B","HC8C","HC8D",
                             "HC8E","HC8F","HC8G","HC8H",
                             "HC8I","HC8J","HC8K","HC8L",
                             "HC8M","HC8N","HC8O","HC8P",
                             "HC8Q","HC8R","HC8S","HC8T",
                             "HC8U","HC8V","HC9A","HC9B",
                             "HC9C","HC9D","HC9E","HC9I",
                             "HC9J","HC9K","HC10","HC11",
                             "HC13","HC15","wscore","HH6"]
        
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

    # print(data[:, 1:35])

    mapper = km.KeplerMapper(verbose=1) # instantiate mapper

    # Makes the lens.I used the sum projection
    lens = mapper.fit_transform(data[:, 1:35],
                                 projection="sum",
                                 scaler=None)
    
    e = 0.0001
    c = 10 # number of open sets to use in the cover
    o = 30 # overlap  (控制覆盖的大小和重叠程度?)

    # run the algorithm to generate the nodes   #原本作者用的是distance2
    nodes = mapper.map(lens, data[:, 1:35],
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
    
    # for node_id, node_data in nodes.items():
    #     print(f"Node ID: {node_id}")
    #     print(f"Node Data: {node_data}")
    #     print()

    # wscores = data[:, -2]
    # for key, values in node_data.items():
    #     for value in values:
    #         wscore = wscores[value]
    #         # 在这里执行您想对每个样本值进行的操作，例如将其存储到另一个数据结构中或进行其他计算。
    #         print("Node value:", value)
    #         print("Corresponding wscore:", wscore)

    node_data = nodes['nodes']
    wscores = data[:, -2]
    for key, values in node_data.items():
        wscore_values = [wscores[value] for value in values]
        # 将节点值替换为对应的 wscore 值
        wscore_values_str = [str(wscore) for wscore in wscore_values]
        replaced_values = {key: wscore_values_str}
        # print(replaced_values)
        # replaced_values[key] = wscore_values_str

        result = {}
        for key, values in replaced_values.items():
                # 将字符串列表转换为浮点数列表
                numeric_values = [float(value) for value in values]
                
                # 获取样本数量
                num_samples = len(numeric_values)

                # 计算均值
                mean_value = np.mean(numeric_values)
                
                # # 计算标准差
                # std_value = np.std(numeric_values)
                # 计算标准差（除以n-1的无偏估计标准差）
                std_value = np.std(numeric_values, ddof=1)
                
                # 计算最大值
                max_value = np.max(numeric_values)
                
                # 计算最小值
                min_value = np.min(numeric_values)
                
                # 存储计算结果到字典中
                result[key] = {
                    'num_samples': num_samples,
                    'mean': mean_value,
                    'std': std_value,
                    'max': max_value,
                    'min': min_value
                }
        print(result)