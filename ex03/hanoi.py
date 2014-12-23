class Action(object):
    
    def __init__(self, my_num, from_peg, from_num, to_peg, to_num):
        self.name = "Move-%s-From-%s-On-%s-To-%s-On-%s" %(my_num, from_peg, from_num, to_peg, to_num)
        
        self.pre = ['Top-%s' %my_num , 'On-%s-%s' %(my_num, from_num), '%s-%s' %(from_peg, my_num)]

        if from_num == "Empty" and to_num == "Empty":
            self.add = ['%s-%s' %(to_peg, my_num)]
            self.dell = ['%s-%s' %(from_peg, my_num)]
        else:
            self.add = ['On-%s-%s' %(my_num, to_num), '%s-%s' %(to_peg, my_num)]    
            self.dell = ['On-%s-%s' %(my_num, from_num), '%s-%s' %(from_peg, my_num)]
             
        if to_num == "Empty":
            self.pre.append('Empty-%s' %to_peg)
            self.dell.append('Empty-%s' %to_peg)
        else:
            self.pre += ['Top-%s' %to_num , '%s-%s' %(to_peg, to_num)]
            self.dell += ['Top-%s' %to_num]
            
        if from_num == "Empty":
            self.add.append("Empty-%s" %from_peg)
        else:
            self.add.append("Top-%s" %from_num)
            self.pre.append('%s-%s' %(from_peg, from_num))
    
    def __str__(self):
        return '\n'.join(['Name: ' + self.name, 
                         'pre: ' + ' '.join(self.pre),
                         'add: ' + ' '.join(self.add),
                         'del: ' + ' '.join(self.dell)])

def createDomainFile(domainFileName, n):
    numbers = list(range(n)) # [0,...,n-1]
    pegs = ['a','b', 'c']
    domainFile = open(domainFileName, 'w') #use domainFile.write(str) to write to domainFile
    
    propositions = []
    actions = []
    
    for num1 in numbers:
        propositions.append("Top-%s" %num1)        
        for peg in pegs:
            propositions.append("%s-%s" %(peg, num1))
        for num2 in ["Empty"] + [k for k in numbers if k > num1]:
            propositions.append('On-%s-%s' %(num1, num2))
    propositions.extend(["Empty-%s" %peg for peg in pegs])
    
    
    for from_peg in pegs:
        for to_peg in pegs:
            if from_peg != to_peg:
                actions.append(Action(n-1, from_peg, "Empty", to_peg, "Empty"))
    
    for my_num in numbers[:-1]:
        for from_peg in pegs:
            for to_peg in pegs:
                if from_peg != to_peg:
                    for from_num in ["Empty"] + [k for k in numbers if k > my_num]:
                        for to_num in ["Empty"] + [k for k in numbers if k > my_num]:
                            if from_num != to_num or from_num == "Empty":
                                actions.append(Action(my_num, from_peg, from_num, to_peg, to_num))
                    
    domainFile.write('Propositions:\n' + ' '.join(propositions) 
                     + "\nActions:\n" + '\n'.join([str(a) for a in actions]))
    domainFile.close()


def createProblemFile(problemFileName, n):
    numbers = list(range(n)) # [0,...,n-1]
    #pegs = ['a','b', 'c']
    problemFile = open(problemFileName, 'w') #use problemFile.write(str) to write to problemFile
    
    init_props = ["Top-0", "a-0", "On-%s-Empty" %(n-1), "Empty-b", "Empty-c"]
    
    for num in numbers[1:]:
        init_props.append("a-%s" %num)
        init_props.append("On-%s-%s" %(num-1, num))
    
    goal_props = ["Top-0", "c-0", "On-%s-Empty" %(n-1), "Empty-b", "Empty-a"]
    
    for num in numbers[1:]:
        goal_props.append("c-%s" %num)
        goal_props.append("On-%s-%s" %(num-1, num))
        
    problemFile.write("Initial state: " + ' '.join(init_props) + 
                      "\nGoal state: " + ' '.join(goal_props))

    problemFile.close()

import sys
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: hanoi.py n')
        sys.exit(2)

    n = int(float(sys.argv[1])) #number of disks
    domainFileName = 'hanoi' + str(n) + 'Domain.txt'
    problemFileName = 'hanoi' + str(n) + 'Problem.txt'

    createDomainFile(domainFileName, n)
    createProblemFile(problemFileName, n)
