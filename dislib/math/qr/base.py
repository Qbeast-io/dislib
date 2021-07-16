from itertools import product
import numpy as np
import warnings

from pycompss.api.api import compss_delete_object
from pycompss.api.constraint import constraint
from pycompss.api.parameter import INOUT, IN, IN_DELETE, COLLECTION_INOUT, Depth, Type, COLLECTION_IN_DELETE, \
    COLLECTION_IN
from pycompss.api.task import task
from pycompss.util.objects.replace import replace

from dislib.data.array import Array, identity, full, eye
from dislib.data.util import compute_bottom_right_shape, pad_last_blocks_with_zeros
from dislib.data.util.base import remove_last_rows, remove_last_columns


def qr_blocked(a: Array, mode='full', overwrite_a=False, save_memory=True):
    """ QR Decomposition (blocked / save memory).

    Parameters
    ----------
    a : ds-arrays
        Input ds-array.
    mode : string
        Mode of the algorithm
        'full' - computes full Q matrix of size m x m and R of size m x n
        'economic' - computes Q of size m x n and R of size n x n
        'r' - computes only R of size m x n
    overwrite_a : bool
        Overwriting the input matrix as R.
    save_memory : bool
        Using the "save memory" algorithm.

    Returns
    -------
    q : ds-array
        only for modes 'full' and 'economic'
    r : ds-array
        for all modes

    Raises
    ------
    ValueError
        If m < n for the provided matrix m x n
        or
        If blocks are not square
        or
        If top left shape is different than regular
        or
        If bottom right block is different than regular
    NotImplementedError
        If a is sparse.
    """

    if a._sparse:
        raise NotImplementedError("Sparse ds-arrays not supported.")

    if mode not in ['full', 'economic', 'r']:
        raise ValueError("Unsupported mode: " + mode)

    if mode == 'economic' and overwrite_a:
        warnings.warn("The economic mode does not overwrite the original matrix. "
                      "Argument overwrite_a is changed to False.", UserWarning)
        overwrite_a = False

    _validate_ds_array(a)

    a_obj = a if overwrite_a else a.copy()

    padded_rows = 0
    padded_cols = 0
    bottom_right_shape = compute_bottom_right_shape(a_obj)
    if bottom_right_shape != a_obj._reg_shape:
        padded_rows = a_obj._reg_shape[0] - bottom_right_shape[0]
        padded_cols = a_obj._reg_shape[1] - bottom_right_shape[1]
        pad_last_blocks_with_zeros(a_obj)

    if mode == "economic":
        q, r = _qr_economic_save_mem(a_obj) if save_memory else _qr_economic(a_obj)
        _undo_padding_economic(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "full":
        q, r = _qr_full_save_mem(a_obj) if save_memory else _qr_full(a_obj)
        _undo_padding_full(q, r, padded_rows, padded_cols)
        return q, r
    elif mode == "r":
        r = _qr_r_save_mem(a_obj) if save_memory else _qr_r(a_obj)
        if padded_cols > 0:
            remove_last_columns(r, padded_cols)
        return r


def _qr_full(r):
    q = identity(r.shape[0], r._reg_shape, dtype=None)

    for i in range(r._n_blocks[1]):
        act_q, r_block = _qr_task(r._blocks[i][i], t=True)
        r.replace_block(i, i, r_block)

        for j in range(r._n_blocks[0]):
            q_block = _dot_task(q._blocks[j][i], act_q, transpose_b=True)
            q.replace_block(j, i, q_block)

        for j in range(i + 1, r._n_blocks[1]):
            r_block = _dot_task(act_q, r._blocks[i][j])
            r.replace_block(i, j, r_block)

        compss_delete_object(act_q)

        sub_q = [[np.array([0]), np.array([0])],
                 [np.array([0]), np.array([0])]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], r_block1, r_block2 = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                r._reg_shape, transpose=True)
            r.replace_block(i, i, r_block1)
            r.replace_block(j, i, r_block2)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r_block1], [r_block2]] = _multiply_blocked(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    r._reg_shape
                )
                r.replace_block(i, k, r_block1)
                r.replace_block(j, k, r_block2)

            for k in range(r._n_blocks[0]):
                [[q_block1, q_block2]] = _multiply_blocked(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    sub_q,
                    r._reg_shape,
                    transpose_b=True
                )
                q.replace_block(k, i, q_block1)
                q.replace_block(k, j, q_block2)

            compss_delete_object(sub_q[0][0])
            compss_delete_object(sub_q[0][1])
            compss_delete_object(sub_q[1][0])
            compss_delete_object(sub_q[1][1])

    return q, r


def _qr_r(r):
    for i in range(r._n_blocks[1]):
        act_q, r_block = _qr_task(r._blocks[i][i], t=True)
        r.replace_block(i, i, r_block)

        for j in range(i + 1, r._n_blocks[1]):
            r_block = _dot_task(act_q, r._blocks[i][j])
            r.replace_block(i, j, r_block)

        compss_delete_object(act_q)

        sub_q = [[np.array([0]), np.array([0])],
                 [np.array([0]), np.array([0])]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], r_block1, r_block2 = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                r._reg_shape, transpose=True)
            r.replace_block(i, i, r_block1)
            r.replace_block(j, i, r_block2)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                [[r_block1], [r_block2]] = _multiply_blocked(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    r._reg_shape
                )
                r.replace_block(i, k, r_block1)
                r.replace_block(j, k, r_block2)

    return r


def _qr_economic(r):
    a_shape = (r.shape[0], r.shape[1])
    a_n_blocks = (r._n_blocks[0], r._n_blocks[1])
    b_size = r._reg_shape

    q = eye(a_shape[0], a_shape[1], b_size, dtype=None)

    act_q_list = []
    sub_q_list = {}

    for i in range(a_n_blocks[1]):
        act_q, r_block = _qr_task(r._blocks[i][i], t=True)
        r.replace_block(i, i, r_block)
        act_q_list.append(act_q)

        for j in range(i + 1, a_n_blocks[1]):
            r_block = _dot_task(act_q, r._blocks[i][j])
            r.replace_block(i, j, r_block)

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = [[np.array([0]), np.array([0])],
                    [np.array([0]), np.array([0])]]

            sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], r_block1, r_block2 = _little_qr(
                r._blocks[i][i], r._blocks[j][i],
                b_size, transpose=True)
            r.replace_block(i, i, r_block1)
            r.replace_block(j, i, r_block2)

            sub_q_list[(j, i)] = sub_q

            # Update values of the row for the value updated in the column
            for k in range(i + 1, a_n_blocks[1]):
                [[r_block1], [r_block2]] = _multiply_blocked(
                    sub_q,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    b_size
                )
                r.replace_block(i, k, r_block1)
                r.replace_block(j, k, r_block2)

    for i in reversed(range(len(act_q_list))):
        for j in reversed(range(i + 1, r._n_blocks[0])):
            for k in range(q._n_blocks[1]):
                [[q_block1], [q_block2]] = _multiply_blocked(
                    sub_q_list[(j, i)],
                    [[q._blocks[i][k]], [q._blocks[j][k]]],
                    b_size,
                    transpose_a=True
                )
                q.replace_block(i, k, q_block1)
                q.replace_block(j, k, q_block2)

            compss_delete_object(sub_q_list[(j, i)][0][0])
            compss_delete_object(sub_q_list[(j, i)][0][1])
            compss_delete_object(sub_q_list[(j, i)][1][0])
            compss_delete_object(sub_q_list[(j, i)][1][1])

        for k in range(q._n_blocks[1]):
            q_block = _dot_task(act_q_list[i], q._blocks[i][k], transpose_a=True)
            q.replace_block(i, k, q_block)

        compss_delete_object(act_q_list[i])

    # removing last rows of r to make it n x n instead of m x n
    remove_last_rows(r, r.shape[0] - r.shape[1])

    return q, r


ZEROS = 0
IDENTITY = 1
OTHER = 2


def _qr_full_save_mem(r):
    b_size = r._reg_shape
    q, q_type = _gen_identity_save_mem(r.shape[0], r.shape[0], r._reg_shape, r._n_blocks[0], r._n_blocks[0])

    r_type = full((r._n_blocks[0], r._n_blocks[1]), (1, 1), OTHER, dtype=np.uint8)

    for i in range(r._n_blocks[1]):
        act_q_type, act_q = _qr_task_save_mem(
            r._blocks[i][i], r_type._blocks[i][i], r._reg_shape, t=True
        )

        for j in range(r._n_blocks[0]):
            _dot_save_mem_a(
                q._blocks[j][i], q_type._blocks[j][i], act_q, act_q_type, b_size, transpose_b=True
            )

        for j in range(i + 1, r._n_blocks[1]):
            _dot_save_mem_b(
                act_q, act_q_type, r._blocks[i][j], r_type._blocks[i][j], b_size
            )

        compss_delete_object(act_q_type)
        compss_delete_object(act_q)

        sub_q_type = [[_type_block_save_mem(OTHER), _type_block_save_mem(OTHER)],
                     [_type_block_save_mem(OTHER), _type_block_save_mem(OTHER)]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = _little_qr_save_mem(
                r._blocks[i][i], r_type._blocks[i][i], r._blocks[j][i], r_type._blocks[j][i],
                r._reg_shape, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                _multiply_blocked_save_mem(
                    sub_q,
                    sub_q_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    r._reg_shape,
                    overwrite_b=True
                )

            for k in range(r._n_blocks[0]):
                _multiply_blocked_save_mem(
                    [[q._blocks[k][i], q._blocks[k][j]]],
                    [[q_type._blocks[k][i], q_type._blocks[k][j]]],
                    sub_q,
                    sub_q_type,
                    r._reg_shape,
                    transpose_b=True,
                    overwrite_a=True
                )

            for row, col in product(range(2), range(2)):
                compss_delete_object(sub_q[row][col])

    return q, r


def _qr_r_save_mem(r):
    b_size = r._reg_shape
    r_type = full((r._n_blocks[0], r._n_blocks[1]), (1, 1), OTHER, dtype=np.uint8)

    for i in range(r._n_blocks[1]):
        act_q_type, act_q = _qr_task_save_mem(
            r._blocks[i][i], r_type._blocks[i][i], r._reg_shape, t=True
        )

        for j in range(i + 1, r._n_blocks[1]):
            _dot_save_mem_b(
                act_q, act_q_type, r._blocks[i][j], r_type._blocks[i][j], b_size
            )

        compss_delete_object(act_q_type)
        compss_delete_object(act_q)

        sub_q_type = [[_type_block_save_mem(OTHER), _type_block_save_mem(OTHER)],
                     [_type_block_save_mem(OTHER), _type_block_save_mem(OTHER)]]

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q = _little_qr_save_mem(
                r._blocks[i][i], r_type._blocks[i][i], r._blocks[j][i], r_type._blocks[j][i],
                r._reg_shape, transpose=True)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, r._n_blocks[1]):
                _multiply_blocked_save_mem(
                    sub_q,
                    sub_q_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    r._reg_shape,
                    overwrite_b=True
                )

            for row, col in product(range(2), range(2)):
                compss_delete_object(sub_q[row][col])

    return r


def _qr_economic_save_mem(r):
    a_shape = (r.shape[0], r.shape[1])
    a_n_blocks = (r._n_blocks[0], r._n_blocks[1])
    b_size = r._reg_shape

    q, q_type = _gen_identity_save_mem(r.shape[0], a_shape[1], b_size, r._n_blocks[0], r._n_blocks[1])

    r_type = full((r._n_blocks[0], r._n_blocks[1]), (1, 1), OTHER, dtype=np.uint8)

    act_q_list = []
    sub_q_list = {}

    for i in range(a_n_blocks[1]):
        act_q_type, act_q = _qr_task_save_mem(
            r._blocks[i][i], r_type._blocks[i][i],
            b_size, t=True
        )
        act_q_list.append((act_q_type, act_q))

        for j in range(i + 1, a_n_blocks[1]):
            _dot_save_mem_b(
                act_q, act_q_type,
                r._blocks[i][j], r_type._blocks[i][j],
                b_size
            )

        # Update values of the respective column
        for j in range(i + 1, r._n_blocks[0]):
            sub_q_type = [[_type_block_save_mem(OTHER), _type_block_save_mem(OTHER)],
                         [_type_block_save_mem(OTHER), _type_block_save_mem(OTHER)]]

            sub_q = _little_qr_save_mem(
                r._blocks[i][i], r_type._blocks[i][i],
                r._blocks[j][i], r_type._blocks[j][i],
                b_size, transpose=True
            )

            sub_q_list[(j, i)] = (sub_q_type, sub_q)

            # Update values of the row for the value updated in the column
            for k in range(i + 1, a_n_blocks[1]):
                _multiply_blocked_save_mem(
                    sub_q,
                    sub_q_type,
                    [[r._blocks[i][k]], [r._blocks[j][k]]],
                    [[r_type._blocks[i][k]], [r_type._blocks[j][k]]],
                    b_size,
                    overwrite_b=True
                )

    for i in reversed(range(len(act_q_list))):
        for j in reversed(range(i + 1, r._n_blocks[0])):
            for k in range(q._n_blocks[1]):
                _multiply_blocked_save_mem(
                    sub_q_list[(j, i)][1],
                    sub_q_list[(j, i)][0],
                    [[q._blocks[i][k]], [q._blocks[j][k]]],
                    [[q_type._blocks[i][k]], [q_type._blocks[j][k]]],
                    b_size,
                    transpose_a=True,
                    overwrite_b=True
                )

            for row, col in product(range(2), range(2)):
                compss_delete_object(sub_q_list[(j, i)][row][col])
            del sub_q_list[(j, i)]

        for k in range(q._n_blocks[1]):
            _dot_save_mem_b(
                act_q_list[i][1], act_q_list[i][0],
                q._blocks[i][k], q_type._blocks[i][k],
                b_size, transpose_a=True
            )

        compss_delete_object(act_q_list[i][0])
        compss_delete_object(act_q_list[i][1])

    # removing last rows of r to make it n x n instead of m x n
    remove_last_rows(r, r.shape[0] - r.shape[1])

    return q, r


def _undo_padding_full(q, r, n_rows, n_cols):
    if n_rows > 0:
        remove_last_rows(q, n_rows)
        remove_last_columns(q, n_rows)
    if n_cols > 0:
        remove_last_columns(r, n_cols)

    remove_last_rows(r, max(r.shape[0] - q.shape[1], 0))


def _undo_padding_economic(q, r, n_rows, n_cols):
    if n_rows > 0:
        remove_last_rows(q, n_rows)
    if n_cols > 0:
        remove_last_columns(r, n_cols)
        remove_last_rows(r, n_cols)
        remove_last_columns(q, n_cols)


def _validate_ds_array(a: Array):
    if a._n_blocks[0] < a._n_blocks[1]:
        raise ValueError("m > n is required for matrices m x n")

    if a._reg_shape[0] != a._reg_shape[1]:
        raise ValueError("Square blocks are required")

    if a._reg_shape != a._top_left_shape:
        raise ValueError("Top left block needs to be of the same shape as regular ones")


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array), a=IN_DELETE)
def _qr_task(a, mode='reduced', t=False):
    from numpy.linalg import qr
    q, r = qr(a, mode=mode)
    if t:
        q = np.transpose(q)
    return q, r


@constraint(computing_units="${computingUnits}")
@task(returns=np.array)
def _dot_task(a, b, transpose_result=False, transpose_a=False, transpose_b=False):
    return _dot(a, b, transpose_result, transpose_a, transpose_b)


def _dot(a, b, transpose_result=False, transpose_a=False, transpose_b=False):
    if transpose_a:
        a = np.transpose(a)
    if transpose_b:
        b = np.transpose(b)
    if transpose_result:
        return np.transpose(np.dot(a, b))
    return np.dot(a, b)


@constraint(computing_units="${computingUnits}")
@task(returns=(np.array, np.array, np.array, np.array, np.array, np.array))
def _little_qr_task(a, b, b_size, transpose=False):
    regular_b_size = b_size[0]
    curr_a = np.bmat([[a], [b]])
    (sub_q, sub_r) = np.linalg.qr(curr_a, mode='complete')
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)
    if transpose:
        return np.transpose(sub_q[0][0]), np.transpose(sub_q[1][0]), np.transpose(sub_q[0][1]), np.transpose(
            sub_q[1][1]), aa, bb
    else:
        return sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1], aa, bb


def _little_qr(a, b, b_size, transpose=False):
    sub_q00, sub_q01, sub_q10, sub_q11, aa, bb = _little_qr_task(a, b, b_size, transpose)
    return sub_q00, sub_q01, sub_q10, sub_q11, aa, bb


@constraint(computing_units="${computingUnits}")
@task(a=IN, b=IN, c=INOUT)
def _multiply_single_block_task(a, b, c, transpose_a=False, transpose_b=False):
    if transpose_a:
        a = np.transpose(a)
    if transpose_b:
        b = np.transpose(b)
    c += (a.dot(b))


def _multiply_blocked(a, b, b_size, transpose_a=False, transpose_b=False):
    if transpose_a:
        new_a = []
        for i in range(len(a[0])):
            new_a.append([])
            for j in range(len(a)):
                new_a[i].append(a[j][i])
        a = new_a

    if transpose_b:
        new_b = []
        for i in range(len(b[0])):
            new_b.append([])
            for j in range(len(b)):
                new_b[i].append(b[j][i])
        b = new_b

    c = []
    for i in range(len(a)):
        c.append([])
        for j in range(len(b[0])):
            c[i].append(np.zeros(b_size))
            for k in range(len(a[0])):
                _multiply_single_block_task(a[i][k], b[k][j], c[i][j], transpose_a=transpose_a, transpose_b=transpose_b)

    return c


def _split_matrix(a, m_size):
    b_size = int(len(a) / m_size)
    split_matrix = [[None for m in range(m_size)] for m in range(m_size)]
    for i in range(m_size):
        for j in range(m_size):
            split_matrix[i][j] = a[i * b_size:(i + 1) * b_size, j * b_size:(j + 1) * b_size]
    return split_matrix


def _gen_identity_save_mem(n, m, b_size, n_size, m_size):
    a = eye(n, m, b_size, dtype=None)
    aux_a = eye(n_size, m_size, (1, 1), dtype=np.uint8)
    return a, aux_a


@constraint(computing_units="${computingUnits}")
@task(a=INOUT, returns=(np.array, np.array))
def _qr_task_save_mem(a, a_type, b_size, mode='reduced', t=False):
    if a_type[0, 0] == OTHER:
        q, r = np.linalg.qr(a, mode=mode)
        q_type = _type_block_save_mem(OTHER)
    elif a_type[0, 0] == ZEROS:
        q = np.identity(b_size[0])
        r = np.zeros(b_size)
        q_type = _type_block_save_mem(IDENTITY)
    else:
        q = np.identity(b_size[0])
        r = np.identity(b_size[0])
        q_type = _type_block_save_mem(IDENTITY)
    if t:
        q = np.transpose(q)

    replace(a, r)
    return q_type, q


def _type_block_save_mem(value):
    return np.full((1, 1), value, np.uint8)


def _empty_block_save_mem(shape):
    return np.zeros(shape)


@constraint(computing_units="${computingUnits}")
@task(a=INOUT, a_type=INOUT)
def _dot_save_mem_a(a, a_type, b, b_type, b_size, transpose_result=False, transpose_a=False, transpose_b=False):
    block_type_new, block_new = _dot_save_mem(a, a_type, b, b_type, b_size, transpose_result, transpose_a, transpose_b)
    replace(a, block_new)
    replace(a_type, block_type_new)


@constraint(computing_units="${computingUnits}")
@task(b=INOUT, b_type=INOUT)
def _dot_save_mem_b(a, a_type, b, b_type, b_size, transpose_result=False, transpose_a=False, transpose_b=False):
    block_type_new, block_new = _dot_save_mem(a, a_type, b, b_type, b_size, transpose_result, transpose_a, transpose_b)
    replace(b, block_new)
    replace(b_type, block_type_new)


def _dot_save_mem(a, a_type, b, b_type, b_size, transpose_result=False, transpose_a=False, transpose_b=False):
    if a_type[0][0] == ZEROS:
        return _type_block_save_mem(ZEROS), _empty_block_save_mem(b_size)
    if a_type[0][0] == IDENTITY:
        if transpose_b and transpose_result:
            return b_type, b
        if transpose_b or transpose_result:
            return _transpose_block_save_mem(b, b_type)

        return b_type, b
    if b_type[0][0] == ZEROS:
        return _type_block_save_mem(ZEROS), _empty_block_save_mem(b_size)
    if b_type[0][0] == IDENTITY:
        if transpose_a:
            a_type, a = _transpose_block_save_mem(a, a_type)
        if transpose_result:
            return _transpose_block_save_mem(a, a_type)

        return a_type, a
    result = _dot(a, b, transpose_result=transpose_result, transpose_a=transpose_a, transpose_b=transpose_b)
    return _type_block_save_mem(OTHER), result


@constraint(computing_units="${computingUnits}")
@task(a=INOUT, type_a=INOUT, b=INOUT, type_b=INOUT,
      returns=(np.array, np.array, np.array, np.array))
def _little_qr_task_save_mem(a, type_a, b, type_b, b_size, transpose=False):
    regular_b_size = b_size[0]
    ent_a = [type_a, a]
    ent_b = [type_b, b]
    for mat in [ent_a, ent_b]:
        if mat[0] == ZEROS:
            mat[1] = np.zeros(b_size)
        elif mat[0] == IDENTITY:
            mat[1] = np.identity(regular_b_size)
    curr_a = np.bmat([[ent_a[1]], [ent_b[1]]])
    (sub_q, sub_r) = np.linalg.qr(curr_a, mode='complete')
    aa = sub_r[0:regular_b_size]
    bb = sub_r[regular_b_size:2 * regular_b_size]
    sub_q = _split_matrix(sub_q, 2)

    replace(a, aa)
    replace(type_a, _type_block_save_mem(OTHER))
    replace(b, bb)
    replace(type_b, _type_block_save_mem(OTHER))

    if transpose:
        return np.transpose(sub_q[0][0]), np.transpose(sub_q[1][0]), np.transpose(sub_q[0][1]), np.transpose(
            sub_q[1][1])
    else:
        return sub_q[0][0], sub_q[0][1], sub_q[1][0], sub_q[1][1]


def _little_qr_save_mem(a, type_a, b, type_b, b_size, transpose=False):
    sub_q00, sub_q01, sub_q10, sub_q11 = _little_qr_task_save_mem(a, type_a, b, type_b, b_size, transpose)
    return [[sub_q00, sub_q01], [sub_q10, sub_q11]]


@constraint(computing_units="${computingUnits}")
@task(c=INOUT, type_c=INOUT)
def _multiply_single_block_task_save_mem(a, type_a, b, type_b, c, type_c, b_size, transpose_a=False, transpose_b=False):
    if type_a[0][0] == ZEROS or type_b[0][0] == ZEROS:
        return

    fun_a = [type_a, a]
    fun_b = [type_b, b]

    if type_c[0][0] == ZEROS:
        replace(c, np.zeros((b_size[0], b_size[1])))
    elif type_c[0][0] == IDENTITY:
        replace(c, np.identity(b_size[0]))

    if fun_a[0][0][0] == IDENTITY:
        if fun_b[0][0][0] == IDENTITY:
            fun_b[1] = np.identity(b_size[0])
        if transpose_b:
            aux = np.transpose(fun_b[1])
        else:
            aux = fun_b[1]
        c += aux
        replace(type_c, _type_block_save_mem(OTHER))
        return

    if fun_b[0][0][0] == IDENTITY:
        if transpose_a:
            aux = np.transpose(fun_a[1])
        else:
            aux = fun_a[1]
        c += aux
        replace(type_c, _type_block_save_mem(OTHER))
        return

    if transpose_a:
        fun_a[1] = np.transpose(fun_a[1])

    if transpose_b:
        fun_b[1] = np.transpose(fun_b[1])

    c += (fun_a[1].dot(fun_b[1]))
    replace(type_c, _type_block_save_mem(OTHER))
    return


def _multiply_blocked_save_mem(a, type_a, b, type_b, b_size, transpose_a=False, transpose_b=False,
                               overwrite_a=False, overwrite_b=False):
    if transpose_a:
        new_a = []
        for i in range(len(a[0])):
            new_a.append([])
            for j in range(len(a)):
                new_a[i].append(a[j][i])
        a = new_a
        new_a_type = []
        for i in range(len(type_a[0])):
            new_a_type.append([])
            for j in range(len(type_a)):
                new_a_type[i].append(type_a[j][i])
        type_a = new_a_type

    if transpose_b:
        new_b = []
        for i in range(len(b[0])):
            new_b.append([])
            for j in range(len(b)):
                new_b[i].append(b[j][i])
        b = new_b
        new_b_type = []
        for i in range(len(type_b[0])):
            new_b_type.append([])
            for j in range(len(type_b)):
                new_b_type[i].append(type_b[j][i])
        type_b = new_b_type

    c = []
    type_c = []
    for i in range(len(a)):
        c.append([])
        type_c.append([])
        for j in range(len(b[0])):
            c[i].append(_empty_block_save_mem(b_size))
            type_c[i].append(_type_block_save_mem(ZEROS))
            for k in range(len(a[0])):
                _multiply_single_block_task_save_mem(
                    a[i][k], type_a[i][k],
                    b[k][j], type_b[k][j],
                    c[i][j], type_c[i][j],
                    b_size, transpose_a=transpose_a, transpose_b=transpose_b)

    if overwrite_a:
        _overwrite_blocks(a, type_a, c, type_c)

    if overwrite_b:
        _overwrite_blocks(b, type_b, c, type_c)


@task(org_block={Type: COLLECTION_INOUT, Depth: 2}, org_block_type={Type: COLLECTION_INOUT, Depth: 2},
      new_block={Type: COLLECTION_IN, Depth: 2}, new_block_type={Type: COLLECTION_IN, Depth: 2})
def _overwrite_blocks(org_block, org_block_type, new_block, new_block_type):
    for i in range(len(org_block)):
        for j in range(len(org_block[i])):
            replace(org_block[i][j], new_block[i][j])
            replace(org_block_type[i][j], new_block_type[i][j])


def _transpose_block_save_mem(a, a_type):
    if a_type[0][0] == ZEROS or a_type[0][0] == IDENTITY:
        return a_type, a
    return _type_block_save_mem(OTHER), np.transpose(a)
