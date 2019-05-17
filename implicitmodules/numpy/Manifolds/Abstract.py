class Manifold(object):
    """
    Abstract class for geometrical descriptors, needs to have the 
      following methods:
          
          -- Cot_to_Vs (m, p, s) with m value of GD, p a cotangent element and
                s a scale (parametrizing a RKHS). It returns a structured dual
                field
         
          -- infinitesimal_action (m, v) with m value of GD, v a structured field.
                Returns the application of m on v
            
          -- dCotDotV(m, p, vs) with m value of GD, p a cotangent element and
                v a structured field. 
                Returns derivative of (p infinitesimal_action(m,v)) wrt m
    
    """
    
    def __init__(self):
        pass

    def infinitesimal_action(self, field):
        """
        Infinitesimal infinitesimal_action : Apply field to self.GD (velocity).
        
        :return result: an object of class manifold with tan filled (self.GD, velocity)
        """
        pass

    def dCotDotV(self):
        pass
