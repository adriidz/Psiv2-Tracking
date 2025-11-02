# Diario de Trabajo
## En este documento los desarolladores del trabajo documentarán el trabajo realizado por cada uno de ellos
### A la hora de usar el "Diario de Trabajo" sugerimos seguir las siguientes indicaciones:
- Añade contenido siempre que vayas a hacer el último commit de tu sesión de trabajo para que todo el mundo sepa que ha sido modificado.
- Usa tu nombre/inicial/acrónimo y fecha de sesion al lado de las tareas que modifiques para que todos sepan quien ha sido el último en trabajar en ellas y pueda preguntar dudas directamente.
- Pese a que las explicaciones no sean malas, siempre es mejor dar detalles sobre el funcionamiento del código en el mismo codigo. Utiliza el diario para datos generales, indicaciones de posibles acciones para el futuro y comentarios sobre los resultados obtenidos.

### Lista de tareas:
- Tracker de vehículos usando YOLOv11n y métricas de IoU y predicción de movimiento (Cristian)
- Contador de vehículos que pasen por una línea horizontal (Miquel)
- Contador de vehículos que pasen por una línea vertical (Miquel)
- TODO: añadir opción en main.py para elegir conteo vertical, horizontal o ambos.
#### Tareas empezadas:
- Tracker de vehículos usando YOLOv11n y métricas de IoU y predicción de movimiento (Cristian). Mejorar tracker con predicción de movimiento, oclusiones, etc (Comentario Miquel)
- Tracker híbrido (Adri)
- Contador de vehículos que pasen por una línea vertical (Miquel)
#### Tareas Completadas:
- Tracker de vehículos usando YOLOv11n y métricas de IoU y predicción de movimiento (Cristian)
- Contador de vehículos que pasen por una línea horizontal (Miquel)
#### Tareas Testeadas:
- Tracker de vehículos usando YOLOv11n y métricas de IoU y predicción de movimiento (Cristian)
- Contador de vehículos que pasen por una línea horizontal (Miquel)
- Contador de vehículos que pasen por una línea vertical (Miquel)
#### Tareas Terminadas
- Tracker de vehículos usando YOLOv11n y métricas de IoU y predicción de movimiento (Cristian)
- Contador de vehículos que pasen por una línea horizontal (Miquel)
### Diario de Trabajo:

#### Sesión Adri - 17/10
He añadido el YOLOv11n con solamente la clase 2 que son los coches para que los identifique cada X frames. Puedes seleccionar si quieres que el video se vea rápido (velocidad de cómputo) o a 1x. Cuando le das a ESC se termina y solo procesa hasta donde has llegado del vídeo.

#### Sesión Miquel - 20/10
Refactorización del código en diferentes funciones para poder trabajar mejor con las funciones. Creado main.py.
Devdiary actualizado con la estructura estándar de el anterior proyecto.

#### Sesión Cristian - 21/10
He añadido un modulo con dos clases, el Tracker que es la clase principal que se encarga de recibir y procesar los resultados del YOLO y la clase Car que es donde el tracker guarda toda la información de los objetos encontrados (de momento bbox e id). 
También cabe decir que he hecho una copia de detection frames que usa el Tracker y que tiene el skip entre frames reducido, ya que esto ayudaba a seguir a los coches (además de que creo que será necesario para cuando ataquemos el movimiento horizontal de los coches).

- Coches con movimiento vertical: Bastantes buenos resultados
- Coches con movimiento horizontal: Los detecta varias veces pero son demasiado rápidos para el IoU puro → Añadir predicciones de movimiento y otras metricas comparativas
- Oculsiones: Pésimo → Ninguna medida añadida todavia al respeco, trabajo a futuro
- Contador Vertical: No implementado
- Contador Horizontal: No implementado
- Tracker: Solo IoU puro

#### Sesión Miquel 22/10
Se ha añadido VehicleCounter.py que contiene la clase VehicleCounter encargada de contar los vehículos que pasan por una línea horizontal o vertical( en este caso solo está la horizontal hecha)
Modifiqué también el process_frames donde crea la clase vehicleCounter y llama a update para contar los vehículos que pasan por la línea. 
Además, dibuja la línea y el contador en el frame. 
Parece hacer el conteo correctamente. Habría que prpbar todos los vídeos para asegurarse. Lo haré mas adelante o si lo haceis vosotros genial.

#### Sesión Adri - 27/10
Hice la clase TrackerHíbrido que usa IoU primero, y si el resultado no es convincente, usa otros criterios a los cuales les asigna un peso (importancia) a cada uno para calcular el nuevo resultado. Es como comentamos con el profe, en forma de 'cascada' solo que también usa pesos. He bajado a 3 el skip y ahora se hace la prueba con el output2.mp4 (el segundo video del campus), donde el contador va regulín. 

P.D. intenté seguir el criterio que solo esté el _match pero no se hasta que punto las funciones de utilities son "universales" asi que bueno... están marcadas cuáles son

### Sesión Miquel 27/10
El contador de arriba a abajo parece funcionar siempre. Hay que ejecutarlo con todos los vídeos y guardar los resultados.
Creado contador izquierda a derecha. Funciona ok, falla cuando hay oclusiones o cuando deja de detectar
vehículos y los vuelve a detectar como uno diferente. Por ende hay que mejorar algo pero no tengo claro
si es el tracker o el contador el problema. Es díficil saberlo

Cambios en el process_frames para crear el contador de izquierda a derecha y llamar a  update.
Faltaría tal vez hacer una opcion en el main para contar solo de arriba abajo, solo de izquierda a derecha o ambos.

### Sesión Miquel 28/10
Probados videos middle, shadow y short. Pequeños ajustes en la línea de conteo para que se vea mejor y esté en la posición correcta usando 
line_start y line_end.

### Sesion Adri - 30/10
Chavales he estado demasiado tiempo intentando hacer ahi fine tuning pero no puedo masssss no se como podríamos mejroar los resultados. Yo lo dejaba así, hablaría de que hemos intentado valorar yolo11n y skip 2 VS yolo11s y skip 5 por ejemplo (mejor yolo a costa de menos frames de info, el nano es mejor la vdd según mis tests), de que hemos intentado aplicar el metodo de cascada, también está el filtro de Kalman pero como que no acaba de funcionar, en vez de usar greedy hay una cosa que es el método hungaro pero como que no se si es relevante del todo, etc. No se son las 2am llevo desde las 8 post trabajo de EGRA a mimir😴

### Sesión Miquel - 02/11
Terminado el readme con la estructura del proyecto, explicación del pipeline, descripción de la detección y del tracking, tipos de trackers usados y resultados obtenidos.
Proyecto finalizado.