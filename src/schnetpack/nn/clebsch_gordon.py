import torch
import schnetpack.nn as snn
import pkg_resources
import itertools as it
import numpy as np

__all__ = [
    'GlebschGordonMatrix']


indx_fn = lambda x: int((x+1)**2) if x >= 0 else 0

class GlebschGordonMatrix(torch.nn.Module):

    """
    Helper class for Glebsch Gordon Coefficients and calculation
    """

    def __init__(self,degrees):

        super().__init__()
        self.register_buffer('degrees', torch.tensor(degrees))
        cg_rep, segment_ids, num_segments = construct_cg_matrix(degrees=self.degrees)

        self.register_buffer('cg_rep', cg_rep)
        self.register_buffer('segment_ids', segment_ids)
        self.register_buffer('num_segments', torch.tensor(num_segments))


    def contraction_fn(self,sphc):
        """
        Args:
            sphc (Tensor): Spherical harmonic coordinates, shape: (n, m_tot)
        Returns: Contraction on degree l=0 for each degree up to l_max, shape: (n, |l|)
        """
        # Element-wise multiplication and squaring
        weighted_sphc = sphc * sphc * self.cg_rep[None, :]  # shape: (n, m_tot)
        
        # Using torch_scatter to perform segment sum
        result = snn.scatter_add(weighted_sphc, self.segment_ids, dim=1, dim_size=self.num_segments)

        return result  # shape: (n, len(degrees))

    def forward(self, chi, idx_j=None, idx_i=None):
        if idx_j is not None and idx_i is not None:
            chi_ij = chi[idx_j] - chi[idx_i]
            contraction_chi_ij = self.contraction_fn(chi_ij)
        else:
            contraction_chi_ij = self.contraction_fn(chi)
        return contraction_chi_ij


def construct_cg_matrix(degrees):

    # get CG coefficients
    cg = torch.diagonal(init_clebsch_gordan_matrix(degrees=torch.tensor(list({0, *degrees})), l_out_max=0), dim1=1, dim2=2)[0]
    # shape: (m_tot**2)
    # if 0 not in degrees:
    #     cg = cg[1:]  # remove degree zero if not in degrees

    cg_rep = []
    #reps = [(d,degrees.count(d)) for d in set(degrees)]

    for d, r in zip(*torch.unique(degrees, return_counts=True)):
        cg_rep += [torch.tile(cg[indx_fn(d - 1): indx_fn(d)], (r,))]

    cg_rep = torch.concatenate(cg_rep)  # shape: (m_tot), m_tot = \sum_l 2l+1 for l in degrees

    segment_ids = torch.tensor(
        [y for y in it.chain(*[[n] * int(2 * degrees[n] + 1) for n in range(len(degrees))])], dtype=torch.long, device=degrees.device)  # shape: (m_tot
    num_segments = len(degrees)

    return cg_rep, segment_ids, num_segments


def load_cgmatrix(degrees):
    stream = pkg_resources.resource_stream(__name__, 'cgmatrix.npz')
    return torch.tensor(np.load(stream)['cg'], dtype=torch.float32) #,device=degrees.device)


def init_clebsch_gordan_matrix(degrees, l_out_max=None):
    """
    Initialize the Clebsch-Gordan matrix (coefficients for the Clebsch-Gordan expansion of spherical basis functions)
    for given ``degrees`` and a maximal output order ``l_out_max`` up to which the given all_degrees shall be
    expanded. Minimal output order is ``min(degrees)``.

    Args:
        degrees (List): Sequence of degrees l. The lowest order can be chosen freely. However, it should
            be noted that all following all_degrees must be exactly one order larger than the following one. E.g.
            [0,1,2,3] or [1,2,3] are valid but [0,1,3] or [0,2] are not.
        l_out_max (int): Maximal output order. Can be both, smaller or larger than maximal order in degrees.
            Defaults to the maximum value of the passed degrees.

    Returns: Clebsch-Gordan matrix,
        shape: (``(l_out_max+1)**2, (l_out_max+1)**2 - (l_in_min)**2, (l_out_max+1)**2 - (l_in_min)**2``)

    """
    if l_out_max is None:
        _l_out_max = max(degrees)
    else:
        _l_out_max = l_out_max

    l_in_max = max(degrees)
    l_in_min = min(degrees)

    offset_corr = indx_fn(l_in_min - 1)
    _cg = load_cgmatrix(degrees)
    # 0:1, 0:9, 0:9
    return _cg[offset_corr:indx_fn(_l_out_max), offset_corr:indx_fn(l_in_max), offset_corr:indx_fn(l_in_max)]