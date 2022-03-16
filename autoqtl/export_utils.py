import deap

def expr_to_tree(ind, pset):
    """Convert the unstructured DEAP pipeline into a tree data-structure.

    Parameters
    ----------
    ind: deap.creator.Individual
       The pipeline that is being exported

    Returns
    -------
    pipeline_tree: list
       List of operators in the current optimized pipeline

    EXAMPLE:
        pipeline:
            "DecisionTreeClassifier(input_matrix, 28.0)"
        pipeline_tree:
            ['DecisionTreeClassifier', 'input_matrix', 28.0]

    """

    def prim_to_list(prim, args):
        if isinstance(prim, deap.gp.Terminal):
            if prim.name in pset.context:
                return pset.context[prim.name]
            else:
                return prim.value

        return [prim.name] + args

    tree = []
    stack = []
    for node in ind:
        stack.append((node, []))
        while len(stack[-1][1]) == stack[-1][0].arity:
            prim, args = stack.pop()
            tree = prim_to_list(prim, args)
            if len(stack) == 0:
                break  # If stack is empty, all nodes should have been seen
            stack[-1][1].append(tree)

    return tree

def generate_pipeline_code(pipeline_tree, operators):
    """Generate code specific to the construction of the sklearn Pipeline.

    Parameters
    ----------
    pipeline_tree: list
        List of operators in the current optimized pipeline

    Returns
    -------
    Source code for the sklearn pipeline

    """
    steps = _process_operator(pipeline_tree, operators)
    pipeline_text = "make_pipeline(\n{STEPS}\n)".format(
        STEPS=_indent(",\n".join(steps), 4)
    )
    return pipeline_text

def _process_operator(operator, operators, depth=0):
    steps = []
    op_name = operator[0]

    if op_name == "CombineDFs":
        steps.append(_combine_dfs(operator[1], operator[2], operators))
    else:
        input_name, args = operator[1], operator[2:]
        autoqtl_op = get_by_name(op_name, operators)

        if input_name != "input_matrix":
            steps.extend(_process_operator(input_name, operators, depth + 1))

        # If the step is an estimator and is not the last step then we must
        # add its guess as synthetic feature(s)
        # classification prediction for both regression and classification
        # classification probabilities for classification if available
        if autoqtl_op.root and depth > 0:
            steps.append(
                "StackingEstimator(estimator={})".format(autoqtl_op.export(*args))
            )
        else:
            steps.append(autoqtl_op.export(*args))
    return steps


def _indent(text, amount):
    """Indent a multiline string by some number of spaces.

    Parameters
    ----------
    text: str
        The text to be indented
    amount: int
        The number of spaces to indent the text

    Returns
    -------
    indented_text

    """
    indentation = amount * " "
    return indentation + ("\n" + indentation).join(text.split("\n"))


def _combine_dfs(left, right, operators):
    def _make_branch(branch):
        if branch == "input_matrix":
            return "FunctionTransformer(copy)"
        elif branch[0] == "CombineDFs":
            return _combine_dfs(branch[1], branch[2], operators)
        elif branch[1] == "input_matrix":  # If depth of branch == 1
            autoqtl_op = get_by_name(branch[0], operators)

            if autoqtl_op.root:
                return "StackingEstimator(estimator={})".format(
                    _process_operator(branch, operators)[0]
                )
            else:
                return _process_operator(branch, operators)[0]
        else:  # We're going to have to make a pipeline
            autoqtl_op = get_by_name(branch[0], operators)

            if autoqtl_op.root:
                return "StackingEstimator(estimator={})".format(
                    generate_pipeline_code(branch, operators)
                )
            else:
                return generate_pipeline_code(branch, operators)

    return "make_union(\n{},\n{}\n)".format(
        _indent(_make_branch(left), 4), _indent(_make_branch(right), 4)
    )

def get_by_name(opname, operators):
    """Return operator class instance by name.

    Parameters
    ----------
    opname: str
        Name of the sklearn class that belongs to a TPOT operator
    operators: list
        List of operator classes from operator library

    Returns
    -------
    ret_op_class: class
        An operator class

    """
    ret_op_classes = [op for op in operators if op.__name__ == opname]

    if len(ret_op_classes) == 0:
        raise TypeError(
            "Cannot found operator {} in operator dictionary".format(opname)
        )
    elif len(ret_op_classes) > 1:
        raise ValueError(
            "Found duplicate operators {} in operator dictionary. Please check "
            "your dictionary file.".format(opname)
        )
    ret_op_class = ret_op_classes[0]
    return ret_op_class

