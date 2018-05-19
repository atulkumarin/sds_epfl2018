import functools
import random
import SVM_pb2


def vec_mul(vec1, vec2):
    '''dot product of two sparse vectors    '''
    result = 0
    if type(vec2) == list:
        for elem in vec2:
            result += elem[1] * vec1.get(elem[0], 0)
    else:
        for key, val in vec2.items():
            result += val * vec1.get(key, 0)

    return result


def ele_vec_mul(vec1, vec2):
    '''element wise product of two sparse vectors   '''
    result = {}

    for elem in vec2:

        result[elem] = vec2[elem] * vec1.get(elem, 0)

    return result


def add_to(vec, sub_vec, res_vector=None):
    '''sum of two sparse vectors    '''

    result = {} if res_vector is None else res_vector

    for key, val in sub_vec.items():

        result[key] = vec.get(key, 0) + val

    return result


def weight_msg_to_dict(msg):
    '''converts the received proto message into a dictionary    '''

    ret = {}

    for entry in msg.entries:

        ret[entry.index] = entry.value

    return ret


def prediction(label, pred):
    '''performs predictions '''

    ok = (pred >= 0 and label == 1) or (pred < 0 and label == -1)

    return 1 if ok else 0


def scalar_vec_mul(scalar, vec):
    '''computes multiplication of a scalar with a vector    '''
    ret = None
    if type(vec) == dict:
        ret = dict([(entry, vec[entry] * scalar) for entry in vec])
    else:
        ret = dict([(entry[0], entry[1] * scalar) for entry in vec])
    return ret


def calc_reg(servicer, sample):
    '''
        INPUT:
        Servicer : SVM SERVICE
        Sample : data vector (list of tupples (id,value) or dict ) 
    '''
    regularizer = {}
    if type(sample) == list:
        for entry in sample:
            elem = servicer.params.get(entry[0], 0)
            regularizer[entry[0]] = elem * elem / servicer.du.get(entry[0], 0)
    else:
        for entry in sample:
            elem = servicer.params.get(entry, 0)
            regularizer[entry] = elem * elem / servicer.du.get(entry, 0)

    return functools.reduce(lambda x, y: x + y, regularizer.values()) * servicer.reg


def calc_reg_grad(servicer, sample):
    regularizer = {}
    if type(sample) == list:
        for entry in sample:
            regularizer[entry[0]] = 2 * servicer.reg * \
                servicer.params.get(entry[0], 0) / servicer.du.get(entry[0], 0)
    else:
        for entry in sample:
            regularizer[entry] = 2 * servicer.reg * \
                servicer.params.get(entry, 0) / servicer.du.get(entry, 0)

    return regularizer


def compute_gradient(servicer, random_indices):
    '''compute gradient for a given training examples   '''


    def compute_sub_gradient(weights, example, label):
        '''compute gradient for a given training examples   '''

        tmp_pt = vec_mul(weights, example)
        acc_pt = prediction(label, tmp_pt)
        tmp_pt = tmp_pt * label

        if (tmp_pt < 1):

            grad_pt = scalar_vec_mul(-label, example)

        else:

            grad_pt = dict()

        reg_grad_pt = calc_reg_grad(servicer, example)

        grad_pt = add_to(grad_pt, reg_grad_pt)


        loss_pt = max(0, 1 - tmp_pt) + calc_reg(servicer, example)

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


def dict_to_weight_msg(dic, label = ''):
    '''converts a dictionary into a proto message   '''

    ret = SVM_pb2.Row(label=label)
    entries = []

    for key, value in dic.items():

        entries.append(SVM_pb2.Entry(index=key, value=value))

    ret.entries.extend(entries)

    return ret


def compute_loss_acc(servicer, data, target):

    loss = 0
    acc = 0

    for idx,d in enumerate(data):
        tmp = vec_mul(servicer.params, d)
        loss += max(0, 1 - tmp) + calc_reg(servicer, d)

        if tmp * target[idx] >= 0:
            acc += 1

    acc /= len(target)
    loss /= len(target)
    return loss, acc


def replace(from_, by, in_place=True):
    buff = from_ if in_place else []
    if not in_place:
        for key, val in from_.items():
            buff.append((key, val))
        buff = dict(buff)

    for key, val in by.items():
        buff[key] = val

    return buff


def load_data(file_path, nb_sample=None):
    '''load the relevant examples into main memory using the seek positions sent by the client  '''
    examples = []
    labels = []
    du = {}
    with open(file_path) as data:
        for sample_ in data:

            if (nb_sample is not None):
                if(len(examples) < nb_sample):
                    if random.random() < 0.2:
                        continue
                else:
                    break

            sample = sample_.split(' ')
            entries = []
            for i in range(2, len(sample) - 1):
                entry = sample[i].split(':')
                entries.append((int(entry[0]), float(entry[1])))

                du[int(entry[0])] = du.get(int(entry[0]), 0) + 1

            examples.append(entries)
            labels.append(int(sample[0].strip()))

    return examples, labels, du
