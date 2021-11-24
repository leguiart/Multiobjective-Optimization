import abc

class ProblemInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'evaluate') and 
                callable(subclass.evaluate) and
                hasattr(subclass, 'stop_criteria') and
                callable(subclass.stop_criteria) and
                hasattr(subclass, 'populate') and
                callable(subclass.populate) and 
                hasattr(subclass, 'generate')and
                callable(subclass.generate)and
                hasattr(subclass, 'extract_solutions')and
                callable(subclass.extract_solutions))


@ProblemInterface.register
class IProblem:
    
    def populate(self, n_individuals : int) -> list:
        """Crea una poblacion inicial de posibles soluciones"""
        pass

    
    def generate(self, pop : list) -> list:
        """Apply evolutionary operators to generate a new set of potential solutions"""
        pass
    
    
    def evaluate(self, X : list) -> list:
        """Evalua las soluciones potenciales del problema"""
        pass

    
    def stop_criteria(self, X_eval : list) -> bool:
        """Regresa si la población ha llegado al criterio de paro"""
        pass

    def extract_solutions(self) -> bool:
        """Regresa si la población ha llegado al criterio de paro"""
        pass


