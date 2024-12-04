from pymatgen.analysis.local_env import BrunnerNN_real,CrystalNN,BrunnerNN_reciprocal,VoronoiNN

import numpy as np
import pandas as pd


from pymatgen.core import Structure, Site,PeriodicSite,Lattice,Species,IStructure
import os
from pymatgen.io.cif import CifFile, CifParser, CifWriter
from pymatgen.io.vasp import *

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
#import alphashape
from scipy.spatial import ConvexHull, Delaunay
from pymatgen.analysis.structure_matcher import StructureMatcher

from itertools import permutations
import itertools

import warnings
warnings.filterwarnings('ignore')
import random

from pymatgen.analysis.chemenv.utils.coordination_geometry_utils import Plane

def calc_in_plane(a,b,c,d):
    mat = np.matrix([[a[0],a[1],a[2],1 ],
                     [b[0],b[1],b[2],1 ],
                     [c[0],c[1],c[2],1 ],
                     [d[0],d[1],d[2],1 ]])
    
    det = np.abs(np.linalg.det(mat))
    
    return dets

def get_all_images():
    '''
    get all jimages 
    '''
    comb = list(permutations(np.array([0,0,0,1,1,1,-1,-1,-1]), 3))
    all_images = []
    for x in comb:
        if x not in all_images:
            all_images.append(x)
    return all_images

# populate_sites 
def populate_sites(structure,jimages):
    '''
    populate sites in different images
    '''
    pop_sites = [] #[[] for site in structure.sites]
    for i in range(len(structure.sites)):
        site = structure.sites[i]
        pop_sites.append(get_site_in_jimages(site,jimages))
    return pop_sites

def get_site_in_jimages(site,jimages):
    '''
    site: PeriodicSite
    jimages: 3x1 array: e.g. [(0,0,1),(0,1,0)]
    return: all the periodic sites in jimages
    '''
    sites_in_jimages = []
    #frac_coords = site.frac_coords
    for jimage in jimages:
        #coords = np.dot(site.lattice.matrix, site.frac_coords+jimage)
        coords_jimage = site.frac_coords+jimage
        sites_in_jimages.append(PeriodicSite(site.species, coords_jimage, site.lattice))
    return sites_in_jimages


def get_plane_from_sites(sites):
    '''
    sites: [PeriodicSites]
    '''
    # randomly select three sites
    selected_sites = random.choices(sites,k=3)
    while points_collinear(selected_sites):
        
        selected_sites = random.choices(sites,k=3)
    coeff = get_plane_from_selected_sites(selected_sites)
    return coeff[0],coeff[1],coeff[2],coeff[3], selected_sites


    
def points_collinear(selected_sites):
    p1 = selected_sites[0].coords
    p2 = selected_sites[1].coords
    p3 = selected_sites[2].coords
    v1 = p2-p1
    v2 = p3-p1
    if np.linalg.norm(np.cross(v1,v2))<0.001:
        return True
    else:
        return False

def get_plane_from_selected_sites(selected_sites):
    a = selected_sites[0].coords
    b = selected_sites[1].coords
    c = selected_sites[2].coords
    return get_plane_from_coords(a,b,c)

def calc_site_plane_distance(A,B,C,D,site):
    p = np.array([A,B,C])
    d = site.coords
    return np.round(np.abs(np.dot(p,d)-D)/np.sqrt(A**2+B**2+C**2),5)

def calc_plane_distance(a,b,c,d):
    '''
    calculate the distance from d to plane formed by a,b,c
    '''
    A,B,C,D = get_plane_from_coords(a,b,c)
    return np.abs(np.dot(p,d)-D)/np.sqrt(A**2+B**2+C**2)

def get_plane_from_coords(a,b,c):
    '''
    calculate the plane equation by a,b,c
    '''
    v1 = c - a
    v2 = b - a
    p = np.cross(v1, v2)
    A, B, C = p
    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    D = np.dot(p, c)
    #print('The equation is {0}x + {1}y + {2}z = {3}'.format(A, B, C, D))
    return A,B,C,D

def get_site_idx(structure_sub,structure_full):
    '''
    Get sites in the structure_full that forms the planar structure_sub. 
    Not working if number of atoms <= 3 on that plane. 
    '''
    site_sub = structure_sub.sites
    site_full = structure_full.sites
    
    sub_full_ids = []
    for i in range(len(site_sub)):
        site_i = site_sub[i]
        for j in range(len(site_full)):
            site_j = site_full[j]
            dist = np.linalg.norm(site_i.coords-site_j.coords)
            if np.round(dist,1)==0:
                #print(dist, site_i,site_j)
                sub_full_ids.append(j)
                
    return sub_full_ids


def get_space_group_info(structure):
    
    try:
        sg_sym = structure.get_space_group_info()[0]
        sg_num = structure.get_space_group_info()[1]
    except:
        #print('ex')
        spg = SpacegroupAnalyzer(structure,symprec=0.0001,angle_tolerance=1)
        sg_sym = spg.get_space_group_symbol()
        sg_num = spg.get_space_group_number()
    
    return sg_sym, sg_num


# 
def sites_in_lattice_frame(structure):
    '''
    tell if a lattice site is within the lattice frame.
    useful to tell which sites are in jimages.
    '''
    s_in_box = []
    for site in structure.sites:
        if point_in_lattice_box(structure.lattice.matrix,site.coords):
            s_in_box.append(site)
            
    return s_in_box

def get_sites_in_plane(structure,pop_sites,A,B,C,D):
    sites_in_plane = []
    sites_plane_dist = []
    for i in range(len(structure.sites)):
        #print(i)
        site = structure.sites[i]
        p_sites = pop_sites[i]
        for p_ in p_sites:
            p_plane_d = calc_site_plane_distance(A,B,C,D,p_)
            if p_plane_d<0.3 and i not in sites_in_plane:
                sites_in_plane.append(i)
                sites_plane_dist.append(p_plane_d)
                
    return sites_in_plane,sites_plane_dist


def get_site_in_out_gph(plane_sites,structure_sub):
    '''
    Given the sites identified in plane: plane_sites
    Find if all sites are included in structure_sub.
    If some sites are not included in the structure_sub, we can be sure: 
    * There is a mismatch between topospro gph and planar sites. 
    (1) This is not a Kagome net without central atom. 
    (2) This may not be a Kagome net with central atom. 
    '''
    sites_in_gph = []
    for site in plane_sites[:]:
        #print(site)
        for site_ in structure_sub.sites:
            if (site.distance(site_))<1e-2 and site not in sites_in_gph:
                #print(site.distance_and_image(site_))
                sites_in_gph.append(site)
    sites_out_gph = [x for x in plane_sites if x not in sites_in_gph] 
    return sites_in_gph, sites_out_gph
def get_plane_from_sites_m(sites):
    '''
    sites: [PeriodicSites]
    select a plane where most (>60%) of the points from sites 
    '''
    # randomly select three sites
    points = [s.coords for s in sites]
    portion_sites_on_plane = 0
    while portion_sites_on_plane <0.6:
        selected_sites = random.choices(sites,k=3)
        while points_collinear(selected_sites):
        
            selected_sites = random.choices(sites,k=3)
        coeff = get_plane_from_selected_sites(selected_sites)
        plane_handle = Plane([coeff[0],coeff[1],coeff[2],-coeff[3]])
        #plane_handle.distances_indices_groups
        
        plane_dist_site_group = plane_handle.distances_indices_groups(points,0.08)
        #print(plane_dist_site_group[0])
        #print(plane_dist_site_group)
        if len(plane_dist_site_group)>1:
            num_sites_group = [x for x in plane_dist_site_group[0] if np.abs(np.round(x,3))<0.05]
            #print(num_sites_group)
            portion_sites_on_plane = len(num_sites_group)/len(points)
            #print(portion_sites_on_plane)
    return coeff[0],coeff[1],coeff[2],coeff[3], selected_sites


def group_angles(angles,delta):
    
    '''
    borrow from pymatgen Plane class. 
    param delta: angle interval for which two angles are considered in the same group.
    '''
    
    
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    indices = [x for x in range(len(sorted_angles))]
    if delta is None:
        delta = 0.05 * np.abs(sorted_angles[indices[-1]])
    iends = [
        ii
        for ii, idist in enumerate(indices, start=1)
        if ii == len(sorted_angles) or (np.abs(sorted_angles[indices[ii]]) - np.abs(sorted_angles[idist]) > delta)
    ]
    #print(sorted_indices)
    grouped_indices = [sorted_indices[iends[ii - 1] : iend] if ii > 0 else sorted_indices[:iend] for ii, iend in enumerate(iends)]
    angle_groups = [angles[xx] for xx in grouped_indices]
    return sorted_angles, angle_groups, grouped_indices



def get_plane_from_sites_m2(sites):
    '''
    sites: [PeriodicSites]
    define a plane function based on the averaged coefficients determined from different sites. 
    '''
    # randomly select three sites
    idx = list(range(len(sites)))
    combination = list(itertools.combinations(idx,3))
    count = 0
    cc = 0
    points = [s.coords for s in sites]
    
    for comb in combination: 
        select_sites = [sites[i] for i in comb]
        if not points_collinear_m2(select_sites):
            
            coeff = get_plane_from_selected_sites(select_sites)
            A,B,C,D=coeff
            my_plane = Plane([A,B,C,-D])
            distance = my_plane.distances_indices_groups(points)
            if np.abs(np.mean(np.round(distance[0],3)))<0.05:
                #print(np.mean(np.round(distance[0],3)))
                cc = np.array(coeff)+cc
                count +=1
    return coeff[0]/count,coeff[1]/count,coeff[2]/count,coeff[3]/count


def get_plane_from_sites_m3(sites):
    '''
    sites: [PeriodicSites]
    define a plane function based on the averaged coefficients determined from different sites. 
    '''
    # find three sets of three sites that have angles between 25 and 155
    idx = list(range(len(sites)))
    combination = list(itertools.combinations(idx,3))
    count = 0
    cc = 0
    points = [s.coords for s in sites]
    
    new_combinations = []
    for comb in combination: 
        select_sites = [sites[i] for i in comb]
        if not points_collinear_m2(select_sites):
            new_combinations.append(comb)
            if len(new_combinations)>=100:
                break
    #print(new_combinations)
    for comb in new_combinations: 
        select_sites = [sites[i] for i in comb]
        coeff = get_plane_from_selected_sites(select_sites)
        A,B,C,D=coeff
        my_plane = Plane([A,B,C,-D])
        distance = my_plane.distances_indices_groups(points)
        if np.abs(np.mean(np.round(distance[0],3)))<0.05:
            #print(np.mean(np.round(distance[0],3)))
            cc = np.array(coeff)+cc
            count +=1
    #print(count)
    return coeff[0]/count,coeff[1]/count,coeff[2]/count,coeff[3]/count

def get_vert_ps(lattice):
    factor = 1.05
    a = lattice[0];b = lattice[1];c = lattice[2]
    combinations = np.array(np.meshgrid(a, b, c)).T.reshape(-1, 3)

    # Find the maximum and minimum values of each component of the combinations
    max_values = np.amax(combinations, axis=0)
    min_values = np.amin(combinations, axis=0)

    # Create an array of the eight vertices of the box
    vertices = np.array([
        [min_values[0], min_values[1], min_values[2]],
        [min_values[0], min_values[1], max_values[2]],
        [min_values[0], max_values[1], min_values[2]],
        [min_values[0], max_values[1], max_values[2]],
        [max_values[0], min_values[1], min_values[2]],
        [max_values[0], min_values[1], max_values[2]],
        [max_values[0], max_values[1], min_values[2]],
        [max_values[0], max_values[1], max_values[2]]
    ])
    
    
    return vertices

def get_vert_ps_m(lattice):
    factor = 1.05
    a = lattice[0];b = lattice[1];c = lattice[2]
    combinations = np.array(np.meshgrid(a, b, c)).T.reshape(-1, 3)

    # Find the maximum and minimum values of each component of the combinations
    max_values = np.amax(combinations, axis=0)
    min_values = np.amin(combinations, axis=0)

    # Create an array of the eight vertices of the box
    vertices = np.array([
        [min_values[0], min_values[1], min_values[2]],
        [min_values[0], min_values[1], max_values[2]],
        [min_values[0], max_values[1], min_values[2]],
        [min_values[0], max_values[1], max_values[2]],
        [max_values[0], min_values[1], min_values[2]],
        [max_values[0], min_values[1], max_values[2]],
        [max_values[0], max_values[1], min_values[2]],
        [max_values[0], max_values[1], max_values[2]]
    ])
    # center of mass
    vertices_COM = np.array([vertices.T[0].mean(),vertices.T[1].mean(),vertices.T[2].mean()])
    vi = vertices - vertices_COM
    new_verts = vi*factor + vertices_COM
    
    return new_verts

def point_in_lattice_box(lattice,test_p):
    #print(lattice)
    #hull_verts = get_vert_ps(lattice)
    hull_verts = get_vert_ps_m(lattice)
    
    #print(hull_verts)
    hull = Delaunay(hull_verts)
    return hull.find_simplex(test_p,tol=1e-3)>=0

def site_in_lattice_box_m(site):
    check = [True for x in site.frac_coords if (x>-0.025)&(x<1.025) ]
    if len(check)==3:
        return True
    else:
        return 
    
def get_plane_sites_clean(plane_sites,
                          plane_sites_clean,
                          jimages,
                          plane_function):
    '''
    Use PBC to get sites on the plane function. 
    plane_function: pymatgen Plane from A,B,C,D
    '''
    plane_st = Structure.from_sites(plane_sites)
    pop_sites = populate_sites(plane_st,jimages)
    #print(pop_sites)
    plane_sites_cleaned = [x for x in plane_sites_clean]
    for i in range(len(plane_sites)):
        site = plane_sites[i]
        if site not in plane_sites_cleaned:
            #print(site)
            for p_site in pop_sites[i]:
                p_site_on_plane = site_on_plane(plane_function,p_site,tol=0.25)
                p_overlapped_known_sites = point_at_known_sites(plane_sites_cleaned,p_site)
                #p_in_lattice = point_in_lattice_box(p_site.lattice.matrix,p_site.coords)
                p_in_lattice = site_in_lattice_box_m(p_site)
                #print(p_in_lattice,p_site_on_plane)
                if p_in_lattice and p_site_on_plane:
                    #print(i)
                    if not p_overlapped_known_sites:
                        #print(p_site)
                        plane_sites_cleaned.append(p_site)
                        break
    # final doublecheck the sites are not overlapping with others
    plane_sites_very_clean = [plane_sites_cleaned[0]]
    
    for i in range(len(plane_sites_cleaned)):
        #print(len(plane_sites_very_clean))
        site_i = plane_sites_cleaned[i]
        add_site = False
        if site_i not in plane_sites_very_clean:
            add_site = True
            for site in plane_sites_very_clean:
                if np.round(np.linalg.norm(site_i.coords - site.coords),2)==0:
                    add_site = False
                    break
        if add_site:
            plane_sites_very_clean.append(site_i)
    plane_sites_pbc = [site for site in plane_sites_very_clean]
    for i in range(len(plane_sites)):
        site = plane_sites[i]
        if site not in plane_sites_very_clean:
            for p_site in pop_sites[i]:
                p_site_on_plane = site_on_plane(plane_function,p_site,tol=0.25)
                p_overlapped_known_sites = point_at_known_sites(plane_sites_pbc,p_site)
                #p_in_lattice = point_in_lattice_box(p_site.lattice.matrix,p_site.coords)
                p_in_lattice = site_in_lattice_box_m(site)
                #print(p_in_lattice,p_site_on_plane)
                if p_in_lattice and p_site_on_plane:
                    #print(i)
                    if not p_overlapped_known_sites:
                        #print(p_site)
                        plane_sites_pbc.append(site)
                        break
    return plane_sites_very_clean,plane_sites_pbc

def get_plane_sites_clean_m2(plane_sites_cleaned):
    plane_sites_very_clean = [plane_sites_cleaned[0]]
    #print(plane_sites_very_clean)
    for i in range(len(plane_sites_cleaned)):
        #print(len(plane_sites_very_clean))
        site_i = plane_sites_cleaned[i]
        add_site = False
        if site_i not in plane_sites_very_clean:
            add_site = True
            for site in plane_sites_very_clean:
                if np.round(np.linalg.norm(site_i.coords - site.coords),2)==0:
                    add_site = False
                    break
        if add_site:
            plane_sites_very_clean.append(site_i)
                
    return plane_sites_very_clean

def site_on_plane(plane_function,site,tol=0.15):
    '''
    return True if site-plane-distance is within tol
    '''
    distance = plane_function.distances_indices_groups([site.coords],0.1)[0][0]
    if np.round(np.abs(distance),3)<tol:
        return True
    else:
        return False
    
def point_at_known_sites(known_sites,site):
    known_coords = [s.coords for s in known_sites]
    distances = [np.round(np.linalg.norm(kc-site.coords),2) for kc in known_coords]
    if 0 in distances:
        return True
    else:
        return False
    
    
    
def r_min_and_max(r_list):
    # 
    return np.min(r_list),np.max(r_list)

def calc_angle(structure,ijk):
    return np.round(structure.get_angle(ijk[0],ijk[1],ijk[2]),3)

def calc_angle(si,sj,sk):
    # angle between sj-si and sj-sk
    v1 = si.coords - sj.coords
    v2 = sk.coords - sj.coords
    return np.round(np.rad2deg(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))),0)
    
#def r_min_and_

    
def get_1nn_on_plane(structure,plane_fun_coeff,factor=1.6):
    # get the 1NN for each site within the radius r_d
    A,B,C,D = plane_fun_coeff
    sites = [site for site in structure.sites if calc_site_plane_distance(A,B,C,D,site)<4e-1]
    site_ids = [i for i in range(len(structure.sites)) 
                if calc_site_plane_distance(A,B,C,D,structure.sites[i])<3e-1]
    # get neighbors up to R = Rx
    # calculate Rx: 
    sites_nn_info = {}
    atomic_distances = np.unique(np.concatenate([[ np.round(site.distance(site_j),4)  
                                                  for site_j in sites if np.round(site.distance(site_j),3) !=0] for site in plane_structure.sites]))
    sort_dis = np.sort(atomic_distances)
    #print(sort_dis)
    # to ensure we cover all cases
    Rx = min(sort_dis[0]*factor,3.5)
    #print(Rx)
    _sites_neighbors = structure.get_all_neighbors((Rx))
    #print(_sites_neighbors)
    #n_sites_n_distances = [[calc_site_plane_distance(A,B,C,D,site) for site in site_s] for site_s in _sites_neighbors]
    #print(n_sites_n_distances)
    n_sites_neighbors = [[site for site in site_s if calc_site_plane_distance(A,B,C,D,site)<4e-1] 
                         for site_s in _sites_neighbors]
    #print([len(x) for x in n_sites_neighbors])
    #print(n_sites_neighbors)
    sites_neighbors = []
    #print(len(_sites_neighbors),len(sites))
    for i in site_ids:
        site = structure.sites[i]
        sites_neighbors.append([site_j for site_j in n_sites_neighbors[i]])
    sites_coordination = [len(x) for x in sites_neighbors]
    return sites, sites_coordination

def points_collinear_m(selected_sites):
    p1 = selected_sites[0].coords
    p2 = selected_sites[1].coords
    p3 = selected_sites[2].coords
    v1 = p2-p1;v1_n = v1/np.linalg.norm(v1)
    v2 = p3-p1;v2_n = v2/np.linalg.norm(v2)
    if np.linalg.norm(np.cross(v1_n,v2_n))<0.01:
        #print(np.linalg.norm(np.cross(v1_n,v2_n)))
        return True
    else:
        return False
    
def points_collinear_m2(selected_sites):
    p1 = selected_sites[0].coords
    p2 = selected_sites[1].coords
    p3 = selected_sites[2].coords
    v1 = p2-p1;#v1_n = v1/np.linalg.norm(v1)
    v2 = p3-p1;#v2_n = v2/np.linalg.norm(v2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle)
    
    if not np.isnan(angle_degrees):
        if angle_degrees < 29 or angle_degrees > 151:
            return True
        else:
            return False
    else:
        #print(angle_degrees)
        return True
def get_plane_from_sites_m(sites):
    '''
    Find the plane with most of the sites. 
    sites: [PeriodicSites]
    '''
    # randomly select three sites
    selected_sites = random.choices(sites,k=3)
    portion_sites_on_plane = 0
    while portion_sites_on_plane<0.65: # this is case-dependent, need a better way
        while points_collinear_m(selected_sites):
            selected_sites = random.choices(sites,k=3)
        coeff = get_plane_from_selected_sites(selected_sites)
        site_plane_distances = [calc_site_plane_distance(coeff[0],coeff[1],coeff[2],coeff[3],site) 
                                for site in sites]
        #print(site_plane_distances)
        sites_on_plane = len([x for x in site_plane_distances if x <4e-1])
        portion_sites_on_plane = sites_on_plane/len(site_plane_distances)
        selected_sites = random.choices(sites,k=3)
    return coeff[0],coeff[1],coeff[2],coeff[3], selected_sites


def get_1nn(structure,
            plane_structure_very_clean,
            plane_func_coeffs,
            NN_method=CrystalNN(x_diff_weight=0.0, porous_adjustment=False )):
    # get the 1NN for each site within the radius r_d
    A,B,C,D = plane_func_coeffs
    plane_handle = Plane([A,B,C,-D])
    sites = plane_structure_very_clean.sites
    sites_nn_info = {}
    B_NN_info_pre = NN_method.get_all_nn_info(structure=structure) 
    B_NN_info = [[] for x in sites]
    
    for i in range(len(sites)):
        nn_site_index = range(len(B_NN_info_pre[i]))
        for j in nn_site_index:
            site_nn_j = B_NN_info_pre[i][j]
            plane_to_point = np.abs(plane_handle.distances_indices_groups([site_nn_j['site'].coords])[0][0])
            #print(plane_to_point)
            if plane_to_point<0.4:
                B_NN_info[i].append(site_nn_j)
        
            
    #sites_coordination = [len(x) for x in B_NN_info]
    #return sites,B_NN_info
    
    for i in range(len(sites)):
        nn_site_index = range(len(B_NN_info[i])) #np.unique([x['site_index'] for x in B_NN_info[i]])
        angles_index_list = [(x[0],i,x[1]) for x in list(itertools.combinations(nn_site_index,2))]
        #print(angles_index_list)
        for j in range(len(B_NN_info[i])):
            B_NN_info[i][j]["distance"] = np.round(sites[i].distance(B_NN_info[i][j]['site']),5)
        #print(len(B_NN_info[i]))
        
        angle_info = [[(B_NN_info[i][x[0]]['site_index'],i,B_NN_info[i][x[2]]['site_index']), 
                       calc_angle(B_NN_info[i][x[0]]['site'],sites[i],B_NN_info[i][x[2]]['site'])] for x in angles_index_list 
                      if calc_angle(B_NN_info[i][x[0]]['site'],sites[i],B_NN_info[i][x[2]]['site']) not in [0,180]]
        
        
        angle_info = [x for x in angle_info if str(x[1])!='nan' ]
        sites_nn_info['{}'.format(str(i))]={"nn_info":B_NN_info[i],
                                            "angle_info":angle_info}
    return sites, sites_nn_info



def calc_angle_m(si,sj,sk):
    # angle between sj-si and sj-sk
    v1 = si.coords - sj.coords
    v2 = sk.coords - sj.coords
    return np.round(np.rad2deg(np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))),3)
    
def calculate_opposite_side_length(angle, side1, side2):
    """
    Takes an angle in degrees and the lengths of the two sides
    forming that angle in a triangle. 
    It returns the length of the side opposite
    to the given angle. 
    """
    
    angle_rad = np.deg2rad(angle)
    opposite_side_length = np.sqrt(side1**2 + side2**2 - 2*side1*side2*np.cos(angle_rad))
    return opposite_side_length

def get_1nn_m(structure,
            plane_structure_very_clean,
            plane_func_coeffs,
            NN_method=CrystalNN(x_diff_weight=0.0, porous_adjustment=False )):
    # get the 1NN for each site within the radius r_d
    A,B,C,D = plane_func_coeffs
    plane_handle = Plane([A,B,C,-D])
    sites = plane_structure_very_clean.sites
    sites_nn_info = {}
    B_NN_info_pre = NN_method.get_all_nn_info(structure=structure) 
    B_NN_info = [[] for x in sites]
    
    for i in range(len(sites)):
        nn_site_index = range(len(B_NN_info_pre[i]))
        for j in nn_site_index:
            site_nn_j = B_NN_info_pre[i][j]
            plane_to_point = np.abs(plane_handle.distances_indices_groups([site_nn_j['site'].coords])[0][0])
            #print(plane_to_point)
            if plane_to_point<0.4:
                B_NN_info[i].append(site_nn_j)
        
            
    #sites_coordination = [len(x) for x in B_NN_info]
    #return sites,B_NN_info
    
    for i in range(len(sites)):
        nn_site_index = range(len(B_NN_info[i])) #np.unique([x['site_index'] for x in B_NN_info[i]])
        angles_index_list = [(x[0],i,x[1]) for x in list(itertools.combinations(nn_site_index,2))]
        #print(angles_index_list)
        for j in range(len(B_NN_info[i])):
            a = sites[i].coords 
            b = B_NN_info[i][j]['site'].coords
            B_NN_info[i][j]["distance"] = np.round(np.linalg.norm(a-b),5) #np.round(sites[i].distance(B_NN_info[i][j]['site']),5)
        #print(len(B_NN_info[i]))
        angle_info = []
        for x in angles_index_list:
            if calc_angle(B_NN_info[i][x[0]]['site'],sites[i],B_NN_info[i][x[2]]['site']) not in [0,180]:
                # (j,i,k)
                j = B_NN_info[i][x[0]]['site_index']
                k = B_NN_info[i][x[2]]['site_index']
                # angle(j,i,k)
                angle = calc_angle_m(B_NN_info[i][x[0]]['site'],sites[i],B_NN_info[i][x[2]]['site'])
                # distances
                dij = B_NN_info[i][x[0]]['distance']
                dik = B_NN_info[i][x[2]]['distance']
                djk = np.round(calculate_opposite_side_length(angle,dij,dik),5)
                angle_info.append([j,i,k, angle, dij,dik,djk] )
                
        angle_info = [x for x in angle_info if str(x[3])!='nan' ]
        sites_nn_info['{}'.format(str(i))]={"nn_info":B_NN_info[i],
                                            "angle_info":angle_info}
    return sites, sites_nn_info

def get_1nn_simple(structure,
            plane_func_coeffs,
            NN_method=CrystalNN(x_diff_weight=0.0, porous_adjustment=False )):
    # get the 1NN for each site within the radius r_d
    A,B,C,D = plane_func_coeffs
    plane_handle = Plane([A,B,C,-D])
    sites = structure.sites
    sites_nn_info = {}
    B_NN_info_pre = NN_method.get_all_nn_info(structure=structure) 
    B_NN_info = [[] for x in B_NN_info_pre]
    for i in range(len(sites)):
        nn_site_index = range(len(B_NN_info_pre[i]))
        for j in nn_site_index:
            site_nn_j = B_NN_info_pre[i][j]
            plane_to_point = np.abs(plane_handle.distances_indices_groups([site_nn_j['site'].coords])[0][0])
            #print(plane_to_point)
            if plane_to_point<0.4:
                B_NN_info[i].append(site_nn_j)
            
    #sites_coordination = [len(x) for x in B_NN_info]
    #return sites,B_NN_info
    
    for i in range(len(sites)):
        nn_site_index = range(len(B_NN_info[i])) #np.unique([x['site_index'] for x in B_NN_info[i]])
        angles_index_list = [(x[0],i,x[1]) for x in list(itertools.combinations(nn_site_index,2))]
        #print(angles_index_list)
        for j in range(len(B_NN_info[i])):
            B_NN_info[i][j]["distance"] = np.round(sites[i].distance(B_NN_info[i][j]['site']),5)
        #print(len(B_NN_info[i]))
        
        angle_info = [[(B_NN_info[i][x[0]]['site_index'],i,B_NN_info[i][x[2]]['site_index']), 
                       calc_angle(B_NN_info[i][x[0]]['site'],sites[i],B_NN_info[i][x[2]]['site'])] for x in angles_index_list 
                      if calc_angle(B_NN_info[i][x[0]]['site'],sites[i],B_NN_info[i][x[2]]['site']) not in [0,180]]
        
        
        angle_info = [x for x in angle_info if str(x[1])!='nan' ]
        sites_nn_info['{}'.format(str(i))]={"nn_info":B_NN_info[i],
                                            "angle_info":angle_info}
    return sites, sites_nn_info

def get_1nn_on_plane_m(structure,plane_fun_coeff,factor=1.6):
    # get the 1NN for each site within the radius r_d
    A,B,C,D = plane_fun_coeff
    site_distances = [calc_site_plane_distance(A,B,C,D,site) for site in structure.sites]
    
    return site_distances

