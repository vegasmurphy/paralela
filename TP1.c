/******************************************************************************
* FILE: mm.c
* LAST REVISED: 04/13/05
******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ROWS 10000
#define COLS ROWS

int main (int argc, char *argv[])
{
    int tid,                /* a task identifier */
        i, ii, j, jj, k, kk, r,s,ths, rc,n,h,
        M,K,BLOCK_SIZE=500;           /* misc */
        

    int alg;

    double **a,           /* matrix A to be multiplied */
			**b,           /* matrix B to be multiplied */
			**c;           /* result matrix C */
           
    	
	a = (double **) malloc (COLS*sizeof(double));
	c = (double **) malloc (COLS*sizeof(double));
	b = (double **) malloc (ROWS*sizeof(double));
	
	for(i=0;i<COLS;i++)
	{
		a[i] = (double *) malloc (ROWS*sizeof(double));
		c[i] = (double *) malloc (ROWS*sizeof(double));      
    }       

	for(i=0;i<ROWS;i++)
	{
		b[i] = (double *) malloc (COLS*sizeof(double));
    }       

    double tini,tfin;


//data initialization
    if (argc>4){
       alg=atoi(argv[1]);
       ths=atoi(argv[2]);
       K=atoi(argv[3]);
       M=atoi(argv[4]);
       BLOCK_SIZE=atoi(argv[5]);
    } else {
       alg=1;
       ths=1;
       K=1;
       M=1;
    }

    for(i=0;i<ROWS;i++){
            for(j=0;j<COLS;j++){
                a[j][i]=1.0;
                c[j][i]=0.0;
            }
    }

    for(i=0;i<ROWS;i++){
            for(j=0;j<COLS;j++){
                b[i][j]=1.0;
            }
    }

    omp_set_num_threads(ths);
   
   
   
    switch (alg){
    case 1:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k)
        for(i=0;i<ROWS;i++){
            for(j=0;j<COLS;j++){
                for(k=0;k<ROWS;k++){
                    c[i][j]=c[i][j]+a[i][k]*b[k][j];
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;
    case 2:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k)
        for(i=0;i<ROWS;i++){
            for(k=0;k<ROWS;k++){
                for(j=0;j<COLS;j++){
                    c[i][j]=c[i][j]+a[i][k]*b[k][j];
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;
	case 3:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k)
         for(j=0;j<COLS;j++){
			for(i=0;i<ROWS;i++){
                for(k=0;k<ROWS;k++){
                    c[i][j]=c[i][j]+a[i][k]*b[k][j];
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;
    case 4:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k)
        for(j=0;j<COLS;j++){
                for(k=0;k<ROWS;k++){
					for(i=0;i<ROWS;i++){
								c[i][j]=c[i][j]+a[i][k]*b[k][j];
							}
				}
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;    
    

    case 5:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k,ii,jj,kk)
        for(i=0;i<=ROWS-BLOCK_SIZE;i=i+BLOCK_SIZE){
            for(k=0;k<=ROWS-BLOCK_SIZE;k=k+BLOCK_SIZE){
                for(j=0;j<=COLS-BLOCK_SIZE;j=j+BLOCK_SIZE){
                    for(ii=0;ii<BLOCK_SIZE;ii++){
                        for(jj=0;jj<BLOCK_SIZE;jj++){
                            for(kk=0;kk<BLOCK_SIZE;kk++){
                                c[i+ii][j+jj]=c[i+ii][j+jj]+a[i+ii][k+kk]*b[k+kk][j+jj];
                            }
                        }
                    }
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;
        case 6:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k,ii,jj,kk)
        for(i=0;i<=ROWS-BLOCK_SIZE;i=i+BLOCK_SIZE){
            for(k=0;k<=ROWS-BLOCK_SIZE;k=k+BLOCK_SIZE){
                for(j=0;j<=COLS-BLOCK_SIZE;j=j+BLOCK_SIZE){
                    for(ii=0;ii<BLOCK_SIZE;ii++){
                    	for(kk=0;kk<BLOCK_SIZE;kk++){
                        	for(jj=0;jj<BLOCK_SIZE;jj++){
                                c[i+ii][j+jj]=c[i+ii][j+jj]+a[i+ii][k+kk]*b[k+kk][j+jj];
                            }
                        }
                    }
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;
    
// El case 7 es para probar el orden de ejecucion kk, ii, jj que no se pudo probar en los algoritmos simples por
// problemas de concurrencia. En este caso, ya no habria problemas porque cada hilo trabaja en una mitad diferente
// de la matriz C.    
        case 7:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k,ii,jj,kk)
		for(i=0;i<=ROWS-BLOCK_SIZE;i=i+BLOCK_SIZE){
            for(j=0;j<=COLS-BLOCK_SIZE;j=j+BLOCK_SIZE){
					for(k=0;k<=ROWS-BLOCK_SIZE;k=k+BLOCK_SIZE){
						for(kk=0;kk<BLOCK_SIZE;kk++){	
							for(ii=0;ii<BLOCK_SIZE;ii++){
								for(jj=0;jj<BLOCK_SIZE;jj++){
										c[i+ii][j+jj]=c[i+ii][j+jj]+a[i+ii][k+kk]*b[k+kk][j+jj];
									}
                        }
                    }
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;
    
        case 8:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
#pragma omp parallel for private (i,j,k,ii,jj,kk)
			for(j=0;j<=COLS-BLOCK_SIZE;j=j+BLOCK_SIZE){
				for(k=0;k<=ROWS-BLOCK_SIZE;k=k+BLOCK_SIZE){			
					for(i=0;i<=ROWS-BLOCK_SIZE;i=i+BLOCK_SIZE){
            			for(ii=0;ii<BLOCK_SIZE;ii++){
							for(kk=0;kk<BLOCK_SIZE;kk++){	
								for(jj=0;jj<BLOCK_SIZE;jj++){
											c[i+ii][j+jj]=c[i+ii][j+jj]+a[i+ii][k+kk]*b[k+kk][j+jj];
										}
                        }
                    }
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;       
     
/* Se prueba el algoritmo 6, que resulto ser el mas rapido, cambiando 
 * el ciclo for a paralelizar
 */    
        case 9:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
        for(i=0;i<=ROWS-BLOCK_SIZE;i=i+BLOCK_SIZE){
            for(k=0;k<=ROWS-BLOCK_SIZE;k=k+BLOCK_SIZE){
#pragma omp parallel for private (j,ii,jj,kk)
                for(j=0;j<=COLS-BLOCK_SIZE;j=j+BLOCK_SIZE){
                    for(ii=0;ii<BLOCK_SIZE;ii++){
                    	for(kk=0;kk<BLOCK_SIZE;kk++){
                        	for(jj=0;jj<BLOCK_SIZE;jj++){
                                c[i+ii][j+jj]=c[i+ii][j+jj]+a[i+ii][k+kk]*b[k+kk][j+jj];
                            }
                        }
                    }
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;  
 
    case 10:
        tini=omp_get_wtime();
        tid = omp_get_thread_num();
		for(i=0;i<=ROWS-BLOCK_SIZE;i=i+BLOCK_SIZE){
#pragma omp parallel for private (j,k,ii,jj,kk)
            for(j=0;j<=COLS-BLOCK_SIZE;j=j+BLOCK_SIZE){
					for(k=0;k<=ROWS-BLOCK_SIZE;k=k+BLOCK_SIZE){
						for(ii=0;ii<BLOCK_SIZE;ii++){	
							for(kk=0;kk<BLOCK_SIZE;kk++){	
								for(jj=0;jj<BLOCK_SIZE;jj++){
										c[i+ii][j+jj]=c[i+ii][j+jj]+a[i+ii][k+kk]*b[k+kk][j+jj];
									}
                        }
                    }
                }
            }
        }
        tfin=omp_get_wtime();
        printf("\nEl computo demoro  %f segs\n",tfin-tini);
    break;    
    break;
    }
    
    
//chequeo de resultado
    for(i=0;i<COLS;i++){
        for(j=0;j<ROWS;j++){
            if(c[i][j]!=ROWS*1.0){
               printf("Error en fila %d col %d %f \n",i,j,c[i][j]);
            }
        }
    }
}
