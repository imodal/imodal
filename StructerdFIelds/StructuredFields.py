#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 06:29:43 2018

@author: barbaragris
"""

import numpy as np
from implicitmodules.src import field_structures as fields
from implicitmodules.src import pairing_structures as pair

class StructuredField(object):
            
    """
    Abstract class for structured field
    
    attributes:
     dic : dictionary with types '0', 'p' or 'm'.
     
     
    Methods : 
        
          -- p_Ximv (vsl, vsr, j) with vsl and vsr 2 structured field supposed
           to be in the same rkhs  and vsl is supposed to hae only '0' and 'p',
           j an integer (0 or 1). 
               If j=0 it returns (p | Xi_m (vsr)) where vsl is 
               parametrized by (m,p), i.e. the inner product between the two
               fields. If j=1 it returns the derivative wrt m
               
          -- Apply : Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
                   
    """
       
    def __init__(self, dim):
        self.dim = dim
        self.dic = dict()
        pass
    
    def fill_dic_from_dic(self, dic):
        self.dic = dict()
        if '0' in dic:
            self.dic['0'] = []
            for (x,p) in dic['0']:
                self.dic['0'].append( (x.copy(), p.copy()) )

        if 'p' in dic:
            for (x,P) in dic['p']:
                self.dic['p'].append( (x.copy(), P.copy()) )
                
                
        if 'm' in dic:
            for (x,P) in dic['m']:
                self.dic['m'].append( (x.copy(), P.copy()) )
                
        
        if 'sig' in dic:
            self.dic['sig'] = dic['sig']



class StructuredField_0(object):
    
    def __init__(self, sigma, dim): #tested
        """
         sigma is the sclae of the rkhs to which the field belongs
         dic is the parametrization of the field
        """
        self.dim = dim
        self.sig = sigma
        self.type = '0'
        self.dic0 = []        
        self.dic = {'0': [], 'sig': self.sig}
    
    
    def copy(self):
        v = StructuredField_0(self.sig, self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_0(self.sig, self.dim)
        (x,p) = self.dic['0'][0]
        v.dic['0'] = [(x.copy(), p.copy())]
            
        return v
        
        
    def fill_fieldparam(self, param):
        """
        param should be a list of two elements: array of points and 
        array of vectors
        """
        self.dic['0'] = [param]
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
    
        return fields.my_VsToV(self.dic, z, j)

       
        
    def p_Ximv(self, vsr, j):
        return pair.my_pSmV(self.dic, vsr.dic, j)
    



class StructuredField_m(object):
    
    def __init__(self, sigma, dim): #
        """
         sigma is the sclae of the rkhs to which the field belongs
         dic is the parametrization of the field
        """
        self.dim = dim
        self.sig = sigma
        self.type = 'm'
        self.dicm = []        
        self.dic = {'m': [], 'sig': self.sig}
    
    
    def copy(self):
        v = StructuredField_m(self.sig, self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_m(self.sig, self.dim)
        (x,P) = self.dic['m'][0]
        v.dic['m'] = [(x.copy(), P.copy())]
            
        return v
        
        
    def fill_fieldparam(self, param):
        """
        param should be a list of two elements: array of points and 
        array of vectors
        """
        self.dic['m'] = [param]
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
    
        return fields.my_VsToV(self.dic, z, j)

       
        
    def p_Ximv(self, vsr, j):
        return pair.my_pSmV(self.dic, vsr.dic, j)
    

class StructuredField_p(object):
    
    def __init__(self, sigma, dim): #
        """
         sigma is the sclae of the rkhs to which the field belongs
         dic is the parametrization of the field
        """
        self.dim = dim
        self.sig = sigma
        self.type = 'p'
        self.dicp = []        
        self.dic = {'p': [], 'sig': self.sig}
    
    
    def copy(self):
        v = StructuredField_p(self.sig, self.dim)
        return v
    
    def copy_full(self):
        v = StructuredField_p(self.sig, self.dim)
        (x,P) = self.dic['p'][0]
        v.dic['p'] = [(x.copy(), P.copy())]
            
        return v
        
        
    def fill_fieldparam(self, param):
        """
        param should be a list of two elements: array of points and 
        array of vectors
        """
        self.dic['p'] = [param]
    
    def Apply(self, z, j):
        """
        Applies the field to points z (or computes the derivative). 
        Needs pre-assigned parametrization of the field in dic
        
        """
    
        return fields.my_VsToV(self.dic, z, j)

       
        
    def p_Ximv(self, vsr, j):
        return pair.my_pSmV(self.dic, vsr.dic, j)
    




    
def sum_structured_fields(fields_list):
    return Summed_field(fields_list)



class Summed_field(object):
    def __init__(self, fields_list):
        self.N_fields = len(fields_list)
        dic = dict()
        dic['0'] = []
        dic['p'] = []
        dic['m'] = []
        ##dic['sig_0'] = []
        #dic['sig_p'] = []
        #dic['sig_m'] = []
        self.dic = dic
        #self.indi_0 = []
        #self.indi_p = []
        #self.indi_m = []
        self.fields_list = fields_list
        self.fill_dic_init()
    
    def copy(self):
        field_list = []
        for i in range (self.N_fields):
            field_list.append(self.fields_list[i].copy())
        return Summed_field(field_list)
    
    def copy_full(self):
        field_list = []
        for i in range (self.N_fields):
            field_list.append(self.fields_list[i].copy_full())
        return Summed_field(field_list)
    
    
    
    def fill_dic_init(self):
        for i in range(self.N_fields):
            fi_dic = self.fields_list[i].dic
            if '0' in fi_dic:
                for (x,P) in fi_dic['0']:
                    self.dic['0'].append( (x, P) )
                    #self.indi_0.append(i)
                    #self.dic['sig_0'].append(fi_dic['sig'])
                    
            if 'p' in fi_dic:
                for (x,P) in fi_dic['p']:
                    self.dic['p'].append( (x, P) )
                    #self.indi_p.append(i)
                    #self.dic['sig_p'].append(fi_dic['sig'])
                    
                    
            if 'm' in fi_dic:
                for (x,P) in fi_dic['m']:
                    self.dic['m'].append( (x, P) )
                    #self.indi_m.append(i)
                    #self.dic['sig_m'].append(fi_dic['sig'])
                    
          
    def fill_dic(self, param):
         for parami, fi in zip(param, self.fields_list):
             fi.fill_dic(parami)
             
         self.fill_dic_init()
         
    def Apply(self, z, j): #tested
        Nz = z.shape[0]
        lsize = ((Nz,2),(Nz,2,2),(Nz,2,2,2))
        djv = np.zeros(lsize[j])
        
        for i in range(self.N_fields):
            djv += self.fields_list[i].Apply(z, j)
        
        return djv
    
    
    
    def p_Ximv(self, vsr, j):
        
        raise NameError('No inner product for summed fields')
    
    

class ZeroField(object):#tested
    def __init__(self, dim):
        self.dim = dim
        self.dic = {'sig':1.}
     
    def copy(self):
        return ZeroField(self.dim)
    
    def copy_full(self):
        return ZeroField(self.dim)
             
    def fill_fieldparam(self, param):
        pass
    
    def Apply(self, z, j):
        return fields.my_VsToV(self.dic, z, j)
        
    
    def p_Ximv(self, vsr, j):
        return pair.my_pSmV(self.dic, vsr.dic, j)
#







