import numpy as np
import csv

for i in range(1, 10):

    with open("A0" + str(i) + "EInstances.csv") as instance_file:
        instance_csv = csv.reader(instance_file)
        raw_instances = []
        for row in instance_csv:
            raw_instances.append(row)

    with open("A0" + str(i) + "ELabels.csv") as instance_file:
        instance_csv = csv.reader(instance_file)
        raw_labels = []
        for row in instance_csv:
            raw_labels.append(row)

    instances = []
    labels = []

    #labels: 0 == no activity, 1 == left, 2 == right

    for label in raw_labels:
        type = label[0]
        pos = int(label[1])
        if type == "768":
            instances.append([raw_instances[pos:pos+500]])
            labels.append(0)
            instances.append([raw_instances[pos + 1500:pos+2000]])
            labels.append(0)
        if type == "769":
            instances.append([raw_instances[pos:pos + 500]])
            labels.append(1)
            instances.append([raw_instances[pos + 250:pos + 750]])
            labels.append(1)
            instances.append([raw_instances[pos + 500:pos + 1000]])
            labels.append(1)
        if type == "770":
            instances.append([raw_instances[pos:pos + 500]])
            labels.append(2)
            instances.append([raw_instances[pos + 250:pos + 750]])
            labels.append(2)
            instances.append([raw_instances[pos + 500:pos + 1000]])
            labels.append(2)
        if type == "771":
            instances.append([raw_instances[pos:pos + 500]])
            labels.append(0)
            instances.append([raw_instances[pos + 250:pos + 750]])
            labels.append(0)
            instances.append([raw_instances[pos + 500:pos + 1000]])
            labels.append(0)
        if type == "772":
            instances.append([raw_instances[pos:pos + 500]])
            labels.append(0)
            instances.append([raw_instances[pos + 250:pos + 750]])
            labels.append(0)
            instances.append([raw_instances[pos + 500:pos + 1000]])
            labels.append(0)

    with open("trial" + str(i) + "_labels.csv", "wb") as labels_out:
        writer = csv.writer(labels_out)
        for label in labels:
            writer.writerow([label])

    with open("trial" + str(i) + "instances.csv", "wb") as instances_out:
        writer = csv.writer(instances_out)
        for instance in instances:
            writer.writerow(instance)
    print str(i) + " done"
