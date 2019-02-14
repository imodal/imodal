from old import GeometricalDescriptors
import old.StructuredFields.SummedFields
from old.DeformationModules import DeformationModules as defmod


class CompoundModules(defmod.DeformationModule):  # tested
    def __init__(self, ModList):
        self.ModList = ModList
        self.NbMod = len(ModList)
        self.init_GD()  # creates self.GD as combination of GDs
        self.init_Cont()  # creates self.Cont
        self.cost = 0.
    
    def init_GD(self):  # tested
        self.GD = old.GeometricalDescriptors.Combine_GD.Combine_GD([Modi.GD for Modi in self.ModList])
    
    def init_Cont(self):  # tested
        self.Cont = [Modi.Cont for Modi in self.ModList]
    
    def copy(self):  # tested
        ModList = []
        for i in range(self.NbMod):
            ModList.append(self.ModList[i].copy())
        return CompoundModules(ModList)
    
    def copy_full(self):  # tested
        ModList = []
        for i in range(self.NbMod):
            ModList.append(self.ModList[i].copy_full())
        return CompoundModules(ModList)
    
    def fill_GD(self, GD):  # tested
        self.GD = GD.copy_full()
        GDlist = GD.GD_list
        for i in range(self.NbMod):
            self.ModList[i].fill_GD(GDlist[i])
        self.init_GD()
    
    def update(self):  # tested
        self.init_GD()
        self.init_Cont()
        for i in range(self.NbMod):
            self.ModList[i].update()
    
    def add_cot(self, GD):
        for i in range(self.NbMod):
            self.ModList[i].add_cot(GD.GD_list[i])
        self.init_GD()
    
    def GeodesicControls_curr(self, GDCot):  # tested0
        for i in range(self.NbMod):
            self.ModList[i].GeodesicControls_curr(GDCot)
        self.init_Cont()
    
    def GeodesicControls(self, GD, GDCot):  # tested0
        Cont = []
        GDlist = GD.GD_list
        for i in range(self.NbMod):
            Cont.append(self.ModList[i].GeodesicControls(GDlist[i], GDCot))
        return Cont
    
    def field_generator_curr(self):  # tested0
        return old.StructuredFields.SummedFields.sum_structured_fields([self.ModList[i].field_generator_curr() for i in range(self.NbMod)])
    
    def field_generator(self, GD, Cont):  # tested
        GDlist = GD.GD_list
        return old.StructuredFields.SummedFields.sum_structured_fields(
            [self.ModList[i].field_generator(GDlist[i], Cont[i]) for i in range(self.NbMod)])
    
    def Cost_curr(self):  # tested0
        for i in range(self.NbMod):
            self.ModList[i].Cost_curr()
        self.cost = sum([self.ModList[i].cost for i in range(self.NbMod)])
    
    def Cost(self, GD, Cont):  # tested0
        GDlist = GD.GD_list
        return sum([self.ModList[i].Cost(GDlist[i], Cont[i]) for i in range(self.NbMod)])
    
    def DerCost_curr(self):  # tested0
        derlist = [self.ModList[i].DerCost_curr() for i in range(self.NbMod)]
        return old.GeometricalDescriptors.Combine_GD.Combine_GD(derlist)
    
    def cot_to_innerprod_curr(self, GDCot, j):  # tested0
        if j == 0:
            out = sum([self.ModList[i].cot_to_innerprod_curr(GDCot, j) for i in range(self.NbMod)])
        if j == 1:
            derlist = [self.ModList[i].cot_to_innerprod_curr(GDCot, j) for i in range(self.NbMod)]
            out = old.GeometricalDescriptors.Combine_GD.Combine_GD(derlist)
        
        return out
