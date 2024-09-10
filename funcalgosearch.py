import numpy as np
##function to calculate angle between three given points--------------------------------------------------------------------------------------
def calc(list):
    pm,p2,p3=list
    pm = np.array(pm)
    p2 = np.array(p2)
    p3 = np.array(p3)
    
    u = p2 - pm
    v = p3 - pm
    
    dot_product = np.dot(u, v)
    
    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)
    
    cos_theta = dot_product / (magnitude_u * magnitude_v)
    
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    theta_rad = np.arccos(cos_theta)
    
    theta_deg = np.degrees(theta_rad)
    
    return (theta_deg)

##==function to find all set of threee poinst with one common point=============================================================================================

def twostepp(thegraph):
        list=[]
        for a in thegraph:
            p1,p2= a
            if thegraph[(p1,p2)]==1:
                for b in thegraph:
                    p3,p4= b
                    
                    if thegraph[(p3,p4)]==1:
                        if p1==p3 :
                            if not [p1,p2,p4] in list:
                                list.append([p1,p2,p4])
                        if p2==p4 :
                            if not [p2,p3,p1] in list:
                                list.append([p2,p3,p1])
                            
            
        return(list)
##function to give the coordinates of the three point names given====================================================   
def give_points_coord(pm,p1,p2,keypoints):
    keypoints=np.squeeze(keypoints)
    x1,y1,z1 = keypoints[pm]
    x2,y2,z2 = keypoints[p1]
    x3,y3,z3 = keypoints[p2]
    list=[[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
    listm=[x1,y1,z1]
    list1=[x2,y2,z2]
    list2=[x3,y3,z3]
    return(list)
##=================================================================================================================

def iterate_give_common_point_names(list_from_twostep,keypoints):
    list_for_angles=[]
    for element in list_from_twostep:
        pm=element[0]
        p1=element[1]
        p2=element[2]
        list=give_points_coord(pm,p1,p2,keypoints)
        angle=calc(list)
        list_for_angles.append(round(angle))
    return(list_for_angles)



