class GeometricalDescriptors(object):
    """
    Abstract class for geometrical descriptors, needs to have the
      following methods:

          -- Cot_to_Vs (m, p, s) with m value of GD, p a cotangent element and
                s a scale (parametrizing a RKHS). It returns a structured dual
                field

          -- Ximv (m, v) with m value of GD, v a structured field.
                Returns the application of m on v

          -- dCotDotV(m, p, vs) with m value of GD, p a cotangent element and
                v a structured field.
                Returns derivative of (p Ximv(m,v)) wrt m

    """
    
    def __init__(self):
        pass


def add_cot(Cot0, Cot1):  # tested0
    """
    adds Cots to self.cot
    Cot needs to be of same type as self.Cot
    """
    sumCot = dict()
    sumCot['0'] = []
    sumCot['x,R'] = []
    if '0' in Cot0:
        if '0' in Cot1:
            N0 = len(Cot0['0'])
            if N0 == len(Cot1['0']):
                for i in range(N0):
                    (x0, p0) = Cot0['0'][i]
                    (x1, p1) = Cot1['0'][i]
                    sumCot['0'].append((x0 + x1, p0 + p1))
            else:
                raise NameError('Not possible to add Cotof different types')
        
        else:
            raise NameError('Not possible to add Cotof different types')
    
    if 'x,R' in Cot0:
        if 'x,R' in Cot1:
            N0 = len(Cot0['x,R'])
            if N0 == len(Cot1['x,R']):
                for i in range(N0):
                    ((x0, R0), (p0, PR0)) = Cot0['x,R'][i]
                    ((x1, R1), (p1, PR1)) = Cot1['x,R'][i]
                    sumCot['x,R'].append(((x0 + x1, R0 + R1), (p0 + p1, PR0 + PR1)))
            else:
                raise NameError('Not possible to add Cotof different types')
        
        else:
            raise NameError('Not possible to add Cotof different types')
    
    return sumCot
