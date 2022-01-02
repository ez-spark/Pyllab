import pyllab

pyllab.get_randomness()
neat = pyllab.neat(2,1)
threads = 10
for i in range(pyllab.GENERATIONS):
    number_genomes = neat.get_number_of_genomes()
    neat.reset_fitnesses()
    for j in range(0,number_genomes,threads):
        n_ff = min(number_genomes-j,threads)
        l = []
        indices = []
        for k in range(j,j+n_ff):
            l.append([0,0])
            indices.append(k)
        out = neat.ff_ith_genomes(l,indices,n_ff)
        for k in range(j,j+n_ff):
            neat.increment_fitness_of_genome_ith(k,1-out[k-j][0])
        l = []
        for k in range(j,j+n_ff):
            l.append([1,0])
        out = neat.ff_ith_genomes(l,indices,n_ff)
        for k in range(j,j+n_ff):
            neat.increment_fitness_of_genome_ith(k,out[k-j][0])
        
        l = []
        for k in range(j,j+n_ff):
            l.append([0,1])
        out = neat.ff_ith_genomes(l,indices,n_ff)
        for k in range(j,j+n_ff):
            neat.increment_fitness_of_genome_ith(k,out[k-j][0])
        
        l = []
        for k in range(j,j+n_ff):
            l.append([1,1])
        out = neat.ff_ith_genomes(l,indices,n_ff)
        for k in range(j,j+n_ff):
            neat.increment_fitness_of_genome_ith(k,1-out[k-j][0])
    neat.generation_run()
    print(neat.get_best_fitness())
    if neat.get_best_fitness() >= 3.5:
        binary = neat.get_neat_in_char()
        print(type(binary))
        print(len(binary))
        print(neat.get_lenght_of_char_neat())
        neat2 = pyllab.neat(2,1,binary)
        
        print(neat2.get_lenght_of_char_neat())
        break
    neat.reset_fitnesses()
print(neat.get_global_innovation_number_connections())
print(neat.get_global_innovation_number_nodes())
