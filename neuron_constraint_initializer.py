import random
import tensorflow as tf


class NeuronConstraintInitializer(tf.keras.initializers.Initializer):

    def get_config(self):  # To support serialization
        return {"input_neuron_list": self.input_neuron_list, "output_neuron_list": self.output_neuron_list,
                "label_name": self.label_name, "rules": self.rules, "layer": self.layer}

    def __init__(self, label_name, input_neuron_list, output_neuron_list, rules, layer):
        self.input_neuron_list = input_neuron_list
        self.output_neuron_list = output_neuron_list
        self.label_name = label_name
        self.rules = rules
        self.layer = layer

    def __call__(self, shape, dtype=None, **kwargs):
        print(self.label_name)
        rule_count = 0

        # first create initial values for weights
        """neuron_weights = []
        for i in range(shape[0]):
            neuron_weight = []
            for j in range(shape[1]):
                neuron_weight.append(random.uniform(-1.0, 1.0))
            neuron_weights.append(neuron_weight)"""
        initializer = tf.keras.initializers.HeNormal()
        values = initializer(shape=shape)
        neuron_weights = values.numpy().tolist()

        if self.layer == "input_layer":
            neuron_weights, rule_count = self.set_neuron_constraints_input_layer(neuron_weights, rule_count)
        if self.layer == "output_layer":
            neuron_weights, rule_count = self.set_neuron_constraints_output_layer(neuron_weights, rule_count)

        print(str(rule_count) + " rule(s) have been instantiated within the neuron weights!")
        tensor = tf.convert_to_tensor(neuron_weights)
        return tensor

    def set_neuron_constraints_input_layer(self, neuron_weights, rule_count):
        relevant_rules, irrelevant_rules = self.get_relevant_rules_input_layer()

        for key, rules in relevant_rules.items():
            for rule in rules.values():
                for r in rule:
                    if r not in irrelevant_rules:
                        outgoing = []
                        count = 0
                        rule_split = r.split("->")
                        neuron_rule = False
                        for k, v in self.input_neuron_list.items():
                            if not neuron_rule:
                                for value in v:
                                    if key == k and key in rule_split[1] and value in rule_split[1].split('\''):
                                        receiving = count
                                        count_2 = 0
                                        for k2, v2 in self.input_neuron_list.items():
                                            for value2 in v2:
                                                if k2 in rule_split[0] and value2 in rule_split[0]:
                                                    outgoing.append(count_2)
                                                    neuron_rule = True
                                                    count_2 += 1
                                                else:
                                                    count_2 += 1
                                        break
                                    else:
                                        count += 1
                            else:
                                break
                        for neuron in outgoing:
                            rule_count += 1
                            if '=' in rule_split[1]:
                                neuron_weights[neuron][receiving] = 3.0
                                neuron_weights[receiving][neuron] = 3.0  # rule can be applied in both directions
                            else:
                                neuron_weights[neuron][receiving] = -3.0
                                neuron_weights[receiving][neuron] = -3.0  # rule can be applied in both directions

        return neuron_weights, rule_count

    def get_relevant_rules_input_layer(self):
        with open(self.rules) as r:
            lines = r.readlines()
            rule = ''
            rules = []
            for line in lines:
                if line == '\n':
                    if not ' and ' in rule:
                        rules.append(rule)
                        rule = ''
                        continue
                    else:
                        rule = ''
                        continue
                else:
                    if rule == '':
                        rule = line
                    else:
                        rule = rule + line
            relevant_rules = {}
            for rule in rules:
                for key, values in self.input_neuron_list.items():
                    if key in rule:
                        if relevant_rules and key in relevant_rules:
                            for value in values:
                                if value in rule:
                                    if relevant_rules[key] and value in relevant_rules[key]:
                                        relevant_rules[key][value].append(rule)
                                    else:
                                        relevant_rules[key][value] = [rule]
                        else:
                            for value in values:
                                if value in rule:
                                    relevant_rules[key] = {value: [rule]}
            irrelevant_rules = []
            for key, values in relevant_rules.items():
                for item in values.values():
                    for rule in item:
                        for k in self.input_neuron_list.keys():
                            if k != key and k in rule.split(' '):
                                rule_split = rule.split('->')
                                if k in rule_split[0] and key in rule_split[1]:
                                    relevant = True
                                if k in rule_split[1] and key in rule_split[0]:
                                    relevant = True
                                break
                            else:
                                relevant = False
                        if not relevant:
                            irrelevant_rules.append(rule)
        return relevant_rules, irrelevant_rules

    def set_neuron_constraints_output_layer(self, neuron_weights, rule_count):
        relevant_rules, irrelevant_rules = self.get_relevant_rules_output_layer()

        for key, rules in relevant_rules.items():
            for rule in rules.values():
                for r in rule:
                    if r not in irrelevant_rules:
                        outgoing = []
                        count = 0
                        rule_split = r.split("->")
                        for k, v in self.output_neuron_list.items():
                            if key == k:
                                for value in v:
                                    if key in rule_split[1] and value in rule_split[1].split('\''):
                                        receiving = count
                                        count_2 = 0
                                        for k2, v2 in self.input_neuron_list.items():
                                            for value2 in v2:
                                                if k2 in rule_split[0] and value2 in rule_split[0]:
                                                    outgoing.append(count_2)
                                                    count_2 += 1
                                                else:
                                                    count_2 += 1
                                        break
                                    else:
                                        count += 1
                        for neuron in outgoing:
                            rule_count += 1
                            if '=' in rule_split[1]:
                                neuron_weights[neuron][receiving] = 3.0
                            else:
                                neuron_weights[neuron][receiving] = -3.0

        return neuron_weights, rule_count

    def get_relevant_rules_output_layer(self):
        with open(self.rules) as r:
            lines = r.readlines()
            rule = ''
            rules = []
            for line in lines:
                if line == '\n':
                    if not ' and ' in rule:
                        rules.append(rule)
                        rule = ''
                        continue
                    else:
                        rule = ''
                        continue
                else:
                    if rule == '':
                        rule = line
                    else:
                        rule = rule + line
            relevant_rules = {}
            label = self.label_name
            for rule in rules:
                rule_split = rule.split("->")
                if label in rule_split[1]:
                    if label in relevant_rules:
                        for value in self.output_neuron_list[label]:
                            if value in rule_split[1].split('\''):
                                if value in relevant_rules[label]:
                                    relevant_rules[label][value].append(rule)
                                else:
                                    relevant_rules[label][value] = [rule]
                    else:
                        for value in self.output_neuron_list[label]:
                            if value in rule_split[1].split('\''):
                                relevant_rules[label] = {value: [rule]}
            irrelevant_rules = []
            for key, values in relevant_rules.items():
                for item in values.values():
                    for rule in item:
                        for k in self.input_neuron_list.keys():
                            rule_split = rule.split('->')
                            if k in rule_split[0]:
                                relevant = True
                                break
                            else:
                                relevant = False
                        if not relevant:
                            irrelevant_rules.append(rule)
        return relevant_rules, irrelevant_rules
