from implicitmodules.numpy.StructuredFields.SupportPoints import SupportPoints


class StructuredField:
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

    def __init__(self):
        pass

    def __call__(self, points, k=0):
        raise NotImplementedError


class SupportStructuredField(StructuredField):
    
    def __init__(self, support, moments, sigma):
        super().__init__()
        self.__support = SupportPoints(support)
        self.__moments = moments
        self.__sigma = sigma  # TODO : put a generic kernel
    
    @property
    def support(self):
        return self.__support
    
    @property
    def moments(self):
        return self.__moments
    
    @property
    def sigma(self):
        return self.__sigma
    
    def __call__(self, points, k):
        raise NotImplementedError
