

def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, 'a') as f:
            print(msg)
            f.write(msg + '\n')

            