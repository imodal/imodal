import numpy as np

from implicitmodules.numpy.StructuredFields.Abstract import StructuredField


def sum_structured_fields(fields_list):
    return Summed_field(fields_list)


class Summed_field(StructuredField):
    def __init__(self, fields_list):
        self.N_fields = len(fields_list)
        self.fields_list = fields_list
    
    def copy(self):
        field_list = []
        for i in range(self.N_fields):
            field_list.append(self.fields_list[i].copy())
        return Summed_field(field_list)
    
    def copy_full(self):
        field_list = []
        for i in range(self.N_fields):
            field_list.append(self.fields_list[i].copy_full())
        return Summed_field(field_list)

    def __call__(self, z, j):  # tested
        Nz = z.shape[0]
        lsize = ((Nz, 2), (Nz, 2, 2), (Nz, 2, 2, 2))
        djv = np.zeros(lsize[j])
        
        for i in range(self.N_fields):
            djv += self.fields_list[i](z, j)
        
        return djv
