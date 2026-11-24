# Sobre este sistema

## Qué es y de dónde salen los datos
Es un monitor de riesgo sísmico relativo. Lee cada hora los sismos reales del USGS (Servicio Geológico de Estados Unidos) de las últimas 96 horas y un modelo de aprendizaje automático estima la probabilidad de un sismo M6.5 o más en los próximos 7 días para 12 zonas del mundo.

## Qué NO hace
No predice terremotos. Compara la situación actual con lo habitual de cada zona. Que una zona tenga un multiplicador alto no significa que vaya a ocurrir un terremoto.

## Cómo leer una alerta
Mira primero la probabilidad: suele ser de uno o pocos por ciento. Luego mira cuántas veces más es de lo normal. Después revisa si hay sismos cercanos (menos de 200 km) o solo actividad lejana o profunda; sin sismos cercanos, la subida es una variación estadística y no actividad local.

## Calidad del modelo
En datos que nunca vio durante el entrenamiento, el modelo mejora a la línea base, pero la mejora es modesta (ROC-AUC 0,64; PR-AUC 0,046), porque el evento es muy raro. Úsalo como indicador orientativo, no como pronóstico.
