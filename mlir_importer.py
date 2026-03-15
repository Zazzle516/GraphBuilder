import numpy as np
import torch_mlir
from torch_mlir.ir import (
    Module, Location,
    InsertionPoint, Operation, Block,
    DenseElementsAttr, IntegerAttr, ArrayAttr, DictAttr, FloatAttr, 
    NoneType, F16Type, F32Type, BF16Type, F64Type, IntegerType, RankedTensorType
)
class MLIRImporter(object):

    def __init__(self,
                 input_shapes: list,
                 output_shapes: list,
                 model_name: str,
                 input_types: list,
                 output_types: list,
                 input_names: list = (),
                 output_names: list = (),
                 do_declare: bool = True):
        """
            input_shape: List[List], put module input shape. ex: [[1, 3, 224, 224]]
            output_shape: List[List], put module output shape. ex: [[1, 1000]]
        """
        assert (len(model_name) > 0)
        self.model_name = model_name
        # NOTE: The context must come from torch_mlir.
        # torch-mlir passes are registered in the __init__ method of torch_mlir.ModuleBuildler
        self.module_builder = torch_mlir.ModuleBuilder()
        self.ctx = self.module_builder.context
        self.loc = Location.unknown(self.ctx)
        self.ctx.__enter__()
        self.loc.__enter__()
        self.input_shapes = input_shapes
        self.output_shapes = output_shapes
        self.num_input = len(self.input_shapes)
        self.num_output = len(self.output_shapes)
        self.F32Type = F32Type.get()
        self.I64Type = IntegerType.get_signless(64)
        self.I32Type = IntegerType.get_signless(32)
        self.insert_point_save_flag = False
        self.mlir_type = {
            "INT8": IntegerType.get_signless(8),
            "UINT8": IntegerType.get_unsigned(8),
            "SINT8": IntegerType.get_signed(8),
            "INT16": IntegerType.get_signless(16),
            "UINT16": IntegerType.get_unsigned(16),
            "INT32": IntegerType.get_signless(32),
            "UINT32": IntegerType.get_unsigned(32),
            "INT64": IntegerType.get_signless(64),  # special
            "UINT64": IntegerType.get_unsigned(64),
            "BOOL": IntegerType.get_signless(1),
            "F64": F64Type.get(),
            "F32": F32Type.get(),
            "F16": F16Type.get(),
            "BF16": BF16Type.get(),
            "DICT": DictAttr.get(),
        }
        if do_declare:
            if len(input_names) == 0:
                input_names = ['anonymous_input_' + str(i) for i in range(self.num_input)]
            if len(output_names) == 0:
                output_names = ['anonymous_output_' + str(i) for i in range(self.num_output)]
            self.declare_func(input_types, output_types, input_names, output_names)

    def __del__(self):
        if self.loc is not None:
            self.loc.__exit__(None, None, None)
            self.loc = None
        if self.ctx is not None:
            self.ctx.__exit__(None, None, None)
            self.ctx = None

    def ArrayAttr(self, data: list, data_type: str = 'INT64'):
        assert (data_type in self.mlir_type)
        if data_type.find("INT") >= 0:
            return ArrayAttr.get([IntegerAttr.get(self.mlir_type[data_type], x) for x in data])
        if data_type == 'F32':
            return ArrayAttr.get([FloatAttr.get_f32(x) for x in data])
        if data_type == 'F64':
            return ArrayAttr.get([FloatAttr.get_f64(x) for x in data])
        if data_type == 'DICT':
            # the data in list has been transformed to DictAttr
            return ArrayAttr.get(data)
        raise RuntimeError("unsupport data type:{}".format(data_type))

    def get_tensor_type(self, output_shapes, type=None):
        if type is None:
            type = self.F32Type
        if output_shapes == []:
            return RankedTensorType.get(tuple(output_shapes), type)
        if output_shapes is None:
            return NoneType.get()
        if isinstance(output_shapes, tuple):
            output_shapes = list(output_shapes)
        assert (isinstance(output_shapes, list))
        assert (len(output_shapes) > 0)
        if not isinstance(output_shapes[0], list) and output_shapes[0] is not None:
            return RankedTensorType.get(tuple(output_shapes), type)
        # multi output
        out_types = []
        for s in output_shapes:
            if s is None:
                out_types.append(NoneType.get())
            else:
                out_types.append(RankedTensorType.get(tuple(s), type))
        return out_types

    def get_value_type(self, value):
        _type = str(value.type)
        _type = _type.split('<')[-1].split('x')[-1].split('>')[0]
        if _type == "f32":
            return self.mlir_type['F32']
        elif _type == "i8":
            return self.mlir_type['INT8']
        elif _type == "ui8":
            return self.mlir_type['UINT8']
        elif _type == "i32" or _type == "si32":
            return self.mlir_type['INT32']
        elif _type == "f16":
            return self.mlir_type['F16']
        elif _type == 'i64':
            return self.mlir_type['INT64']
        elif _type == 'i1':
            return self.mlir_type['BOOL']
        else:
            raise RuntimeError("No support {}".format(_type))

    def get_dense_array_attr(self, value):
        return DenseElementsAttr.get(np.ascontiguousarray(np.array(value)))

    def reconfig_insert_point(self, block):
        self.insert_point_back = self.insert_point \
            if not self.insert_point_save_flag else self.insert_point_back
        self.insert_point = InsertionPoint(block)
        self.insert_point_save_flag = True

    def restore_insert_point(self):
        self.insert_point = self.insert_point_back
        self.insert_point_save_flag = False

    def create_return_op(self, operands):
        return_op = Operation.create("func.return", operands=operands, results=[])
        self.insert_point.insert(return_op)
        return return_op

    def create_yield_op(self, operands):
        yield_op = Operation.create("top.Yield", operands=operands, results=[])
        self.insert_point.insert(yield_op)
        return yield_op

    def create_block_at_start(self, region, arg_types):
        return Block.create_at_start(region, arg_types=arg_types)

    def print_module(self):
        mlir_format = self.mlir_module.operation.get_asm(enable_debug_info=True,large_elements_limit=16)
        return mlir_format

    def declare_func(self, input_types: list, output_types: list, input_names: list, output_names: list):
        assert (len(input_types) == self.num_input)
        assert (len(output_types) == self.num_output)
        assert (len(input_names) == self.num_input)
        assert (len(output_names) == self.num_output)
        self.input_types = list()
        self.input_op_types = list()
        self.output_types = list()
        for _shape, _type in zip(self.input_shapes, input_types):
            self.input_op_types.append(RankedTensorType.get(_shape, self.F32Type))
            if isinstance(_type, str):
                self.input_types.append(RankedTensorType.get(_shape, self.mlir_type[_type]))
            else:
                self.input_types.append(RankedTensorType.get(_shape, _type))
        for _shape, _type in zip(self.output_shapes, output_types):
            t = _type
            if isinstance(_type, str):
                t = self.mlir_type[_type]
            self.output_types.append(self.get_tensor_type(_shape, t))
        args_txt = str()
        for _idx, _type in enumerate(self.input_types):
            args_txt += "%args{}: {} loc(unknown)".format(_idx, _type.__str__())
            if (_idx + 1) < self.num_input:
                args_txt += ", "

        output_txt = str()
        for _idx, _type in enumerate(self.output_types):
            output_txt += _type.__str__()
            if (_idx + 1) < self.num_output:
                output_txt += ", "
        result_types = output_txt
        result_var_name = "%1"
        if self.num_output > 1:
            output_txt = "({})".format(output_txt)
            result_types = output_txt[1:-1]
            result_var_name = ",".join([f"%1#{var_id}" for var_id in range(self.num_output)])
        main_func = """
            module attributes {{torch.debug_module_name=\"{name}\"}} {{
                func.func @\"{name}\"({args}) -> {output} attributes {{input_names = [{input_names}], output_names = [{output_names}]}} {{
                    %1:{last_output_num} = \"Placeholder.Op\"() : () -> {output}
                    return {result_var} : {result_types}
                }} loc(unknown)
            }} loc(unknown)
        """.format(name=self.model_name,
                   input_names=','.join(['\"' + n + '\"' for n in input_names]),
                   output_names=','.join(['\"' + n + '\"' for n in output_names]),
                   args=args_txt,
                   output=output_txt,
                   last_output_num=self.num_output,
                   result_var=result_var_name,
                   result_types=result_types)
        self.ctx.allow_unregistered_dialects = True
        self.mlir_module = Module.parse(main_func, self.ctx)
        self.ctx.allow_unregistered_dialects = False
        self.func = self.mlir_module.body.operations[0]
        self.entry_block = self.func.regions[0].blocks[0]
        self.insert_point = InsertionPoint(self.entry_block)
        self.none_op = self.entry_block.operations[0].operation.results[0]
        # remove Placeholder.Op and return Op.
        # These operations are placeholders and are only used to generate a legal MLIR code.
        self.entry_block.operations[1].operation.erase()
        self.entry_block.operations[0].operation.erase()

        self.func_args = list()
        for i in self.entry_block.arguments:
            self.func_args.append(i)

    def get_module(self):
        return self.mlir_module
