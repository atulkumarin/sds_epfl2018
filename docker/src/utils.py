import functools
import random
import SVM_pb2


def vec_mul(vec1, vec2):
    '''dot product of two sparse vectors    '''
    result = 0
    if type(vec2) == list:
        for elem in vec2:
            tmp = vec1.get(elem[0], None)
            if tmp is not None:
                result += elem[1] * tmp
    else:
        for key, val in vec2.items():
            tmp = vec1.get(elem[0], None)
            if tmp is not None:
                result += elem[1] * tmp

    return result


def ele_vec_mul(vec1, vec2):
    '''element wise product of two sparse vectors   '''
    result = {}

    for elem in vec2:
        tmp = vec1.get(elem, None)
        if tmp is not None:
            result[elem] = vec2[elem] * tmp

    return result


def add_to(vec, sub_vec, inplace=False):
    '''sum of two sparse vectors    '''

    result = vec if inplace else vec.copy()

    for key, val in sub_vec.items():
        result[key] = result.get(key, 0) + val

    return result


def weight_msg_to_dict(msg):
    '''converts the received proto message into a dictionary    '''

    ret = {}

    for entry in msg.entries:

        ret[entry.index] = entry.value

    return ret


def prediction(label, pred):
    '''performs predictions '''

    ok = label * pred

    return 1 if (ok >= 0) else 0


def scalar_vec_mul(scalar, vec):
    '''computes multiplication of a scalar with a vector    '''
    ret = None
    if type(vec) == dict:
        ret = dict([(entry, vec[entry] * scalar) for entry in vec])
    else:
        ret = dict([(entry[0], entry[1] * scalar) for entry in vec])
    return ret


def calc_reg(params, du, reg, sample):
    '''
        INPUT:
        Servicer : SVM SERVICE
        Sample : data vector (list of tupples (id,value) or dict ) 
    '''
    regularizer = {}
    if type(sample) == list:
        for entry in sample:
            elem = params.get(entry[0], None)
            if elem is not None:
                regularizer[entry[0]] = elem * elem / du.get(entry[0], 1)
    else:
        for entry in sample:
            elem = params.get(entry, None)
            if elem is not None:
                regularizer[entry] = elem * elem / du.get(entry, 1)

    if len(regularizer):
        return functools.reduce(lambda x, y: x + y, regularizer.values()) * reg
    else:
        return 0


def calc_reg_grad(servicer, sample):
    ''' Compute the regularization gradient '''
    regularizer = {}
    if type(sample) == list:
        for entry in sample:
            w_component = servicer.params.get(entry[0], None)
            if w_component is not None:
                regularizer[entry[0]] = 2 * servicer.reg * \
                    w_component / \
                    servicer.du.get(entry[0], 1)
    else:
        for entry in sample:
            w_component = servicer.params.get(entry[0], None)
            if w_component is not None:
                regularizer[entry] = 2 * servicer.reg * \
                    w_component / servicer.du.get(entry, 1)

    return regularizer


def compute_gradient(servicer, random_indices):
    '''compute gradient for a batch  '''

    def compute_sub_gradient(weights, example, label):
        '''compute gradient for a given training example   '''

        tmp_pt = vec_mul(weights, example)
        acc_pt = prediction(label, tmp_pt)
        tmp_pt = tmp_pt * label

        if (tmp_pt < 1):

            grad_pt = scalar_vec_mul(-label, example)

        else:

            grad_pt = dict()

        reg_grad_pt = calc_reg_grad(servicer, example)

        grad_pt = add_to(grad_pt, reg_grad_pt)

        loss_pt = max(0, 1 - tmp_pt) + calc_reg(servicer.params,
                                                servicer.du, servicer.reg, example)

        return grad_pt, loss_pt, acc_pt

    batch_grad = {}
    final_acc = 0
    final_loss = 0
    for i in random_indices:
        data_pt = servicer.data[i]
        target_label = servicer.target[i]
        grad_pt, loss, acc = compute_sub_gradient(
            servicer.params, data_pt, target_label)
        final_acc += acc
        final_loss += loss
        batch_grad = add_to(batch_grad, grad_pt)

    final_loss = final_loss / len(random_indices)
    final_acc = final_acc / len(random_indices)
    return batch_grad, final_acc, final_loss


def dict_to_weight_msg(dic, label=''):
    '''converts a dictionary into a proto message   '''

    ret = SVM_pb2.Row(label=label)
    entries = []

    for key, value in dic.items():

        entries.append(SVM_pb2.Entry(index=key, value=value))

    ret.entries.extend(entries)

    return ret


def compute_loss_acc(servicer, data, target):
    ''' Compute mean loss and accuracy for the given data '''
    loss = 0
    acc = 0

    for idx, d in enumerate(data):
        loss_pt, acc_pt = compute_loss_acc_pt(
            d, target[idx], servicer.params, servicer.du, servicer.reg)
        loss += loss_pt
        acc += acc_pt

    acc /= len(target)
    loss /= len(target)
    return loss, acc


def compute_loss_acc_pt(data, target, weight, du, reg):
    ''' Compute loss and accuracy for one data point '''
    tmp = vec_mul(weight, data)
    acc = prediction(target, tmp)
    tmp *= target
    loss = max(0, 1 - tmp) + calc_reg(weight, du, reg, data)
    return loss, acc


def replace(from_, by, in_place=True):
    '''Replacing function NOT USED '''
    buff = from_ if in_place else []
    if not in_place:
        for key, val in from_.items():
            buff.append((key, val))
        buff = dict(buff)

    for key, val in by.items():
        buff[key] = val

    return buff


def load_data_real(file_path, label_file, nb_sample_per_class=None, proba_sample=None, class_='CCAT', compute_du=True):
    '''load the relevant examples into main memory  '''
    examples = []
    labels = []
    du = None
    if compute_du:
        du = {}
    pos = 0
    neg = 0
    labels_data_file = None

    with open(label_file) as labels_file_:
        labels_data_file = labels_file_.readlines()
    base = int(labels_data_file[0].split(' ')[1])
    end = int(labels_data_file[-1].split(' ')[1])
    labels_all = [-1] * (end - base + 1)

    for line in labels_data_file:
        line = line.split(' ')
        idx = int(line[1]) - base
        if (line[0] == class_):
            labels_all[idx] = 1

    del labels_data_file
    with open(file_path) as data:
        # idx = 0
        for sample_ in data:
            sample = sample_.split(' ')
            label = labels_all[int(sample[0]) - base]
            if (nb_sample_per_class is not None):
                if(pos < nb_sample_per_class or neg < nb_sample_per_class):

                    neg_thresh = (label == -1 and neg >= nb_sample_per_class)
                    pos_thresh = (label == 1 and pos >= nb_sample_per_class)

                    if (random.random() > proba_sample) or neg_thresh or pos_thresh:
                        continue
                else:
                    break
            if label == 1:
                pos += 1
            else:
                neg += 1
            entries = []
            for i in range(2, len(sample)):
                entry = sample[i].split(':')
                entries.append((int(entry[0]), float(entry[1])))
                if compute_du:
                    du[int(entry[0])] = du.get(int(entry[0]), 0) + 1

            examples.append(entries)
            labels.append(label)

    return examples, labels, du, base, labels_all


def compute_du(file_path):
    '''Compute the du for every component on the given dataset'''
    du = {}
    idx = 0
    with open(file_path) as data:
        for sample_ in data:
            sample = sample_.split(' ')
            idx += 1
            for i in range(2, len(sample)):
                entry = sample[i].split(':')
                du[int(entry[0])] = du.get(int(entry[0]), 0) + 1
    return du


def preprocess_line(line, labels_all, base):
    ''' Preprocess a sample on the dataset file'''
    line = line.split(' ')
    try:
        label = labels_all[int(line[0]) - base]
    except IndexError:
        print('[ERROR] line_nb:{} base:{} len:{}'.format(
            int(line[0]), base, len(labels_all)))
    entries = []
    for i in range(2, len(line)):
        entry = line[i].split(':')
        entries.append((int(entry[0]), float(entry[1])))
    return entries, label
