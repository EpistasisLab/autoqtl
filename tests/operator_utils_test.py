import sys

from numpy import var
sys.path.append("C:/Users/ghosha/.vscode/autoqtl")
import autoqtl

from autoqtl.operator_utils import AUTOQTLOperatorClassFactory, Operator, ARGType
from autoqtl.config.regressor import regressor_config_dict

config_dict = regressor_config_dict

for key in sorted(config_dict.keys()):
    op_class, arg_types = AUTOQTLOperatorClassFactory(
        key, config_dict[key], BaseClass=Operator, ArgBaseClass=ARGType, verbose=0
    )

    print(op_class)
    print(arg_types)
    # print(op_class.__dict__)
    print("NEXT")