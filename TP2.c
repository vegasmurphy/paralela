/*Comunicaciones punto a punto. Programar un envio de mensajes punto a punto de forma tal de formar un ring entre 
todos los nodos del cluster.

Cada nodo le envia al siguiente un entero, y el ultimo vuelve al primer nodo. 
Utilizar un numero arbitrario de nodos.
Programar usando mensajes bloqueantes, de nodo par a impar, y luego al reves, de forma tal de evitar la "inanicion".
Programar tambien usando mensajes no bloqueantes simultaneos entre todos los nodos.
*/


#include <stdio.h>
#include "mpi.h"

int main(int argc, char* argv[])
{
	int  numtasks, my_rank, rc, next, prev, buf_recv; 
   	char hostname[MPI_MAX_PROCESSOR_NAME];
   	MPI_Status status;
      MPI_Request request;

   	rc = MPI_Init(&argc,&argv);
   	if (rc != MPI_SUCCESS) 
   	{
     	printf ("Error al iniciar el programa.\n");
     	MPI_Abort(MPI_COMM_WORLD, rc);
    }

   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);
  
  
   // Determino el nodo siguiente y previo correspondiente a cada uno
   if(my_rank == 0)
   		prev = numtasks - 1;
   else
   		prev = my_rank - 1;
   if(my_rank == numtasks-1)
   		next = 0;
   else
   		next = my_rank + 1;

      /*
   	// Envio y recepcion de mensaje
   	if(!my_rank%2)
   	{
   		MPI_Send(&my_rank,1,MPI_INT,next,1, MPI_COMM_WORLD);
   		MPI_Recv (&buf_recv,1,MPI_INT,prev,1,MPI_COMM_WORLD,&status);
   	}
   	else
   	{
   		MPI_Recv (&buf_recv,1,MPI_INT,prev,1,MPI_COMM_WORLD,&status);
   		MPI_Send(&my_rank,1,MPI_INT,next,1, MPI_COMM_WORLD);
   	}

   	printf ("Soy = %d y recibi de = %d \n", my_rank, buf_recv);

      MPI_Barrier (MPI_COMM_WORLD);

      */
      MPI_Isend(&my_rank,1,MPI_INT,next,1,MPI_COMM_WORLD,&request);
      MPI_Recv (&buf_recv,1,MPI_INT,prev,1,MPI_COMM_WORLD,&status);

      printf ("Soy = %d y recibi de = %d \n", my_rank, buf_recv);


   MPI_Finalize();	
}
