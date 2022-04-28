cimport Pyllab
from Pyllab import *


cdef class neat:
    cdef Pyllab.neat* _neat
    cdef bint _keep_parents
    cdef int _species_threshold
    cdef int _initial_population
    cdef int _generations
    cdef float _percentage_survivors_per_specie
    cdef float _connection_mutation_rate
    cdef float _new_connection_assignment_rate
    cdef float _add_connection_big_specie_rate
    cdef float _add_connection_small_specie_rate
    cdef float _add_node_specie_rate
    cdef float _activate_connection_rate
    cdef float _remove_connection_rate
    cdef int _children
    cdef float _crossover_rate
    cdef int _saving
    cdef int _limiting_species
    cdef int _limiting_threshold
    cdef int _max_population
    cdef int _same_fitness_limit
    cdef float _age_significance
    cdef int _inputs
    cdef int _outputs
    
    def __cinit__(self,int inputs, int outputs, bytes neat_as_byte = None, bint keep_parents = True, int species_threshold = Pyllab.SPECIES_THERESHOLD, int initial_population = Pyllab.INITIAL_POPULATION,
                  int generations = Pyllab.GENERATIONS, float percentage_survivors_per_specie = Pyllab.PERCENTAGE_SURVIVORS_PER_SPECIE, float connection_mutation_rate = Pyllab.CONNECTION_MUTATION_RATE,
                  float new_connection_assignment_rate = Pyllab.NEW_CONNECTION_ASSIGNMENT_RATE, float add_connection_big_specie_rate = Pyllab.ADD_CONNECTION_BIG_SPECIE_RATE,
                  float add_connection_small_specie_rate = Pyllab.ADD_CONNECTION_SMALL_SPECIE_RATE, float add_node_specie_rate = Pyllab.ADD_NODE_SPECIE_RATE, float activate_connection_rate = Pyllab.ACTIVATE_CONNECTION_RATE,
                  float remove_connection_rate = Pyllab.REMOVE_CONNECTION_RATE, int children = Pyllab.CHILDREN, float crossover_rate = Pyllab.CROSSOVER_RATE, int saving = Pyllab.SAVING,
                  int limiting_species = Pyllab.LIMITING_SPECIES, int limiting_threshold = Pyllab.LIMITING_THRESHOLD, int max_population = Pyllab.MAX_POPULATION, int same_fitness_limit = Pyllab.SAME_FITNESS_LIMIT,
                  float age_significance = Pyllab.AGE_SIGNIFICANCE):
        check_int(initial_population)
        check_int(generations)
        check_int(children)
        check_int(inputs)
        check_int(outputs)
        check_int(saving)
        check_int(limiting_species)
        check_int(limiting_threshold)
        check_int(max_population)
        check_int(same_fitness_limit)
        check_int(species_threshold)
        check_float(percentage_survivors_per_specie)
        check_float(connection_mutation_rate)
        check_float(new_connection_assignment_rate)
        check_float(add_connection_big_specie_rate)
        check_float(add_connection_small_specie_rate)
        check_float(add_node_specie_rate)
        check_float(activate_connection_rate)
        check_float(remove_connection_rate)
        check_float(crossover_rate)
        check_float(age_significance)
        
        self._keep_parents = keep_parents
        self._species_threshold = species_threshold
        self._initial_population = initial_population
        self._generations = generations
        self._percentage_survivors_per_specie = percentage_survivors_per_specie
        self._connection_mutation_rate = connection_mutation_rate
        self._new_connection_assignment_rate = new_connection_assignment_rate
        self._add_connection_big_specie_rate = add_connection_big_specie_rate
        self._add_connection_small_specie_rate = add_connection_small_specie_rate
        self._add_node_specie_rate = add_node_specie_rate
        self._activate_connection_rate = activate_connection_rate
        self._remove_connection_rate = remove_connection_rate
        self._children = children
        self._crossover_rate = crossover_rate
        self._saving = saving
        self._limiting_species = limiting_species
        self._limiting_threshold = limiting_threshold
        self._max_population = max_population
        self._same_fitness_limit = same_fitness_limit
        self._age_significance = age_significance
        self._inputs = inputs
        self._outputs = outputs
        cdef char* s
        
        if inputs == 0 or outputs == 0:
            print("Error: either you pass inputs and outputs > 0 or a neat as char characters!")
            exit(1)
        
        if neat_as_byte != None:
            s = neat_as_byte
            
            self._neat = Pyllab.init_from_char(&s[0], inputs,outputs,initial_population,species_threshold,max_population,generations, 
                                               percentage_survivors_per_specie, connection_mutation_rate, new_connection_assignment_rate, add_connection_big_specie_rate, 
                                               add_connection_small_specie_rate, add_node_specie_rate, activate_connection_rate, remove_connection_rate, children, crossover_rate, 
                                               saving, limiting_species, limiting_threshold, same_fitness_limit, keep_parents, age_significance)
            
            
        else:
            self._neat = Pyllab.init(inputs,outputs,initial_population,species_threshold,max_population,generations, 
                                     percentage_survivors_per_specie, connection_mutation_rate, new_connection_assignment_rate, add_connection_big_specie_rate, 
                                     add_connection_small_specie_rate, add_node_specie_rate, activate_connection_rate, remove_connection_rate, children, crossover_rate, 
                                     saving, limiting_species, limiting_threshold, same_fitness_limit, keep_parents, age_significance)
        if self._neat is NULL:
            raise MemoryError()
            
        
    def __dealloc__(self):
        Pyllab.free_neat(self._neat)
    
    def get_best_fitness(self):
        n = Pyllab.best_fitness(self._neat)
        return n
        
    def generation_run(self):
        Pyllab.neat_generation_run(self._neat)
    
    def reset_fitnesses(self):
        Pyllab.reset_fitnesses(self._neat)
        
    
    def reset_fitness_ith_genome(self, int index):
        check_int(index)
        Pyllab.reset_fitness_ith_genome(self._neat, index)
        
    def ff_ith_genomes(self, inputs, indices, int n_genomes):
        value_i = None
        value_e = None
        check_int(n_genomes)
        v_indices = vector_is_valid_int(indices)
        
        if v_indices.ndim != 1:
            print("Error: your dimension must be 1")
            exit(1)
        check_size(v_indices, n_genomes)
        n = get_first_dimension_size(inputs)
        if n < n_genomes:
            n_genomes = n
        if n_genomes > self.get_number_of_genomes():
            print("Error: there are not so many genomes")
            exit(1)
        
        cdef npc.ndarray[npc.npy_float32, ndim=2, mode = 'c'] i_buff
        cdef npc.ndarray[npc.npy_int, ndim=1, mode = 'c'] e_buff
        cdef float** ret = NULL
        for j in range(n_genomes):
            check_size(inputs[j], self._inputs)
            if j == 0:
                value_i = np.array([vector_is_valid(inputs[j])])
            else:
                value_i = np.append(value_i, np.array([vector_is_valid(inputs[j])]),axis=0)
        value_e = vector_is_valid_int(v_indices)
        e_buff = np.ascontiguousarray(value_e,dtype=np.intc)
        i_buff = np.ascontiguousarray(value_i,dtype=np.float32)
        cdef float** i = <float**>malloc(sizeof(float*)*n_genomes)
        cdef int* e = <int*>&e_buff[0]
        for j in range(n_genomes):
            i[j] = &i_buff[j, 0]
        ret = Pyllab.feed_forward_iths_genome(self._neat, &i[0], &e[0], n_genomes)
        l = []
        for j in range(n_genomes):
            l.append(from_float_to_list(ret[j], self._outputs))
            free(ret[j])
        free(ret)
        free(i)
        return np.array(l, dtype='float')
        
    def ff_ith_genome(self,inputs, int index):
        check_int(index)
        value_i = vector_is_valid(inputs)
        check_size(value_i,self._inputs)
        cdef npc.ndarray[npc.npy_float32, ndim=1, mode = 'c'] i_buff
        i_buff = np.ascontiguousarray(value_i,dtype=np.float32)
        cdef float* i = &i_buff[0]
        cdef float* output = Pyllab.feed_forward_ith_genome(self._neat, &i_buff[0], index)
        nd_output = from_float_to_ndarray(output,self._outputs)
        free(output)
        return nd_output
   
    def increment_fitness_of_genome_ith(self,int index, float increment):
        check_int(index)
        check_float(increment)
        Pyllab.increment_fitness_of_genome_ith(self._neat,index,increment)
    
    def get_global_innovation_number_nodes(self):
        return Pyllab.get_global_innovation_number_nodes(self._neat)
   
    def get_global_innovation_number_connections(self):
        return Pyllab.get_global_innovation_number_connections(self._neat)
   
    def get_fitness_of_ith_genome(self, int index):
        check_int(index)
        return Pyllab.get_fitness_of_ith_genome(self._neat, index)
   
    def get_lenght_of_char_neat(self):
        return Pyllab.get_lenght_of_char_neat(self._neat)
   
    def get_neat_in_char(self):
        cdef char* c = Pyllab.get_neat_in_char_vector(self._neat)
        s = c[:self.get_lenght_of_char_neat()]
        free(c)
        return s
   
    def get_number_of_genomes(self):
        return Pyllab.get_number_of_genomes(self._neat)
   
    def save_ith_genome(self, int index, int n):
        check_int(n)
        check_int(index)
        Pyllab.save_ith_genome(self._neat, index, n)
