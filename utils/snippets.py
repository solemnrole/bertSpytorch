# -*- coding:utf-8 -*-
import six
import numpy as np

_open_ = open
is_py2 = six.PY2

if not is_py2:
    basestring = str


def to_array(*args):
    """批量转numpy的array
    """
    results = [np.array(a) for a in args]
    if len(args) == 1:
        return results[0]
    else:
        return results


def is_string(s):
    """判断是否是字符串
    """
    return isinstance(s, basestring)


def convert_to_unicode(text, encoding='utf-8', errors='ignore'):
    """字符串转换为unicode格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, str):
            text = text.decode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text

class open:
    """模仿python自带的open函数，主要是为了同时兼容py2和py3
    """

    def __init__(self, name, mode='r', encoding=None, errors='ignore'):
        if is_py2:
            self.file = _open_(name, mode)
        else:
            self.file = _open_(name, mode, encoding=encoding, errors=errors)
        self.encoding = encoding
        self.errors = errors

    def __iter__(self):
        for l in self.file:
            if self.encoding:
                l = convert_to_unicode(l, self.encoding, self.errors)
            yield l

    def read(self):
        text = self.file.read()
        if self.encoding:
            text = convert_to_unicode(text, self.encoding, self.errors)
        return text

    def write(self, text):
        if self.encoding:
            text = convert_to_str(text, self.encoding, self.errors)
        self.file.write(text)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

def convert_to_str(text, encoding='utf-8', errors='ignore'):
    """字符串转换为str格式（假设输入为utf-8格式）
    """
    if is_py2:
        if isinstance(text, unicode):
            text = text.encode(encoding, errors=errors)
    else:
        if isinstance(text, bytes):
            text = text.decode(encoding, errors=errors)
    return text


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs)



class DataGenerator():
    '''此方法只适用于data能够一次性加载到内存的情况'''
    def __init__(self,
                 batch_size,
                 data,
                 buffer_size=None,
                 ):
        self.data=data
        self.batch_size=batch_size
        if hasattr(self.data,'__len__'):
            self.steps=len(self.data)//self.batch_size
            if len(self.data)%self.batch_size!=0:
                self.steps+=1
        else:
            self.steps=None
        self.buffer_size=buffer_size or batch_size*1000

    def sample(self,random=False):
        """采样函数，每个样本同时返回一个is_end标记,出自bert4keras
        """
        if random:
            if self.steps is None:
                def generator():
                    caches,isfull=[],False
                    for d in self.data:
                        if len(caches)==self.buffer_size:
                            isfull=True
                        caches.append(d)
                        if isfull:
                            i=np.random.randint(len(caches))
                            yield caches.pop(i)
                    while caches:
                        i = np.random.randint(len(caches))
                        yield caches.pop(i)
            else:
                def generator():
                    indices=list(range(len(self.data)))
                    np.random.shuffle(indices)
                    for i in indices:
                        yield self.data[i]
            data=generator()
        else:
            data=iter(self.data)

        d_current=next(data)
        for d_next in data:
            yield False,d_current
            d_current=d_next
        yield True,d_current
