class DeformationModule(object):
    """
    Abstract class for deformation module
    
    
    Attributes: type of GD
                dimension of controls ?
                
                
    Methods: 
        -- GeodesicControls (v_s, m) where v_s is in V^\ast, m a GD,
              returns the corresponding geodesic control
        -- FieldGenerator (m, h) where m is a GD, h a control. It returns
              a structured field in V
        -- Cost (m, h) where m is a GD, h a control. It returns the cost
              (scalar)
        -- DerCost (m, h) where m is a GD, h a control. It returns the
              derivative of the cost wrt m
        -- fill_GD(GD) fills self.GD with value (uses GD structure)
        -- update that stores useful functions (after filling of GD)
    
    For each method, there is a "_curr" version which supposes that GD are 
        known (and necessary functions computes)
    
    """
    
    def add_GD(self, GDCot):
        self.GD.add_GD(GDCot)
    
    def add_tan(self, GDCot):
        self.GD.add_tan(GDCot)
    
    def add_cotan(self, GDCot):
        self.GD.add_cotan(GDCot)
    
    def mult_GD_scal(self, s):
        self.GD.mult_GD_scal(s)
    
    def mult_tan_scal(self, s):
        self.GD.mult_tan_scal(s)
    
    def mult_cotan_scal(self, s):
        self.GD.mult_cotan_scal(s)
    
    def add_speedGD(self, GDCot):
        self.GD.add_speedGD(GDCot)
    
    def sum_GD(self, GD0, GD1):
        self.GD.sum_GD(GD0, GD1)
