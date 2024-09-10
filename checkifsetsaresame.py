list1 =[0, 0, 20, 100, 41, 20, 0, 0, 98, 46, 0, 0, 0, 0, 100, 98, 0, 0, 59, 41, 46, 59, 0, 0, 25, 0, 0, 99, 60, 0, 0, 0, 0, 26, 0, 0, 25, 99, 0, 0, 40, 60, 40, 0, 0, 26, 0, 0, 9, 9, 0, 0, 83, 83, 0, 0, 0, 0, 0, 0, 0, 0]
liti=[0, 0, 9, 147, 172, 9, 0, 0, 151, 163, 0, 0, 0, 0, 147, 151, 0, 0, 37, 172, 163, 37, 0, 0, 21, 0, 0, 53, 14, 0, 0, 0, 0, 34, 0, 0, 21, 53, 0, 0, 44, 14, 44, 0, 0, 34, 0, 0, 78, 78, 0, 0, 107, 107, 0, 0, 0, 0, 0, 0, 0, 0]
def turnlistdict(list):
    count=0
    dicti={}
    for x in list:
        dicti[count]=x
        count=count+1
        
    if len(dicti)<62:
        count1=len(dicti)
        while count1<62:
            dicti[count1]=0
            count1=count1+1
    return(dicti)


def similarity_percentage(num1, num2):
    # Avoid division by zero if both numbers are zero
    if num1 == 0 and num2 == 0:
        return 100.0

    # Calculate the absolute difference between the numbers
    difference = abs(num1 - num2)
    
    # Calculate the average of the two numbers
    average = (abs(num1) + abs(num2)) / 2

    # Calculate the similarity percentage
    similarity = 100 * (1 - (difference / average))

    # Return the similarity percentage, ensuring it is within [0, 100]
    return max(0, min(similarity, 100))

def dictcheck(dict1,dict2,percsims):
    count=0
    list=[]
    for x in dict1:
       
        y=similarity_percentage(dict1[count], dict2[count])
        
        if y>=percsims:
            list.append(1)
        else:
            list.append(0)
        count=count+1
    return(list)

def checklistsum(list,minperctrues):
    sum=0
    for x in list:
        sum=x+sum
    perc=(sum/62)*100
    if perc>=minperctrues:
        return(True)
    else:
        return(False)
    
def final(l1,l2,percsum,minperctrues):
    d1=turnlistdict(l1)
    d2=turnlistdict(l2)
    l3=dictcheck(d1,d2,percsum)
    x=checklistsum(l3,minperctrues)
    return(x)
    
x=final(list1,liti,100,100)
print(x)


