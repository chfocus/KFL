
def agg_func(knowlwdges):
    for [label, know_list] in knowlwdges.items():
        if len(know_list) > 1:
            know = 0 * know_list[0].data
            for i in know_list:
                know += i.data
            knowlwdges[label] = know / len(know_list)
        else:
            knowlwdges[label] = know_list[0]

    return knowlwdges

def knowledge_aggregation(local_knows_list): # dict
    agg_knows_label = dict()
    for idx in local_knows_list:
        local_knows = local_knows_list[idx] # dict
        for label in local_knows.keys():
            if label in agg_knows_label:
                agg_knows_label[label].append(local_knows[label])
            else:
                agg_knows_label[label] = [local_knows[label]]

    for [label, know_list] in agg_knows_label.items():
        if len(know_list) > 1:
            know = 0 * know_list[0].data
            for i in know_list:
                know += i.data
            agg_knows_label[label] = [know / len(know_list)]
        else:
            agg_knows_label[label] = [know_list[0].data]

    return agg_knows_label

