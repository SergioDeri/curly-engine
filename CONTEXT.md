# Contexto

Asistente de soporte al cliente de una tienda online de electrónica.

## Glosario

**Cliente** (`Customer`): persona que compró en la tienda y consulta al asistente.

**Pedido** (`Order`): compra de un producto realizada por un Cliente, identificada por un número de pedido. Solo el Cliente que presenta el número de pedido junto con el email de la compra puede consultarlo.

**Estado del pedido**: en preparación, en camino, entregado o cancelado.

**Devolución** (`Return`): solicitud de un Cliente para devolver un Pedido completo, sujeta a la Política de devoluciones. Hay como máximo una por Pedido.

**Garantía** (`Warranty`): cobertura del fabricante o de la tienda sobre un producto después del plazo de Devolución.

**Política** (`Policy`): documento oficial de la tienda (envíos, devoluciones, garantía, FAQ). Es la única fuente válida para responder preguntas de conocimiento.

**Conversación** (`Thread`): intercambio continuo entre un Cliente y el asistente; conserva su historial entre mensajes.

**Supervisor**: agente que decide qué agente especializado atiende cada paso de una Conversación.

**Agente de Conocimiento** (`Knowledge agent`): responde preguntas usando únicamente las Políticas, citando sus fuentes.

**Agente de Pedidos** (`Orders agent`): consulta Pedidos y gestiona Devoluciones.

**Respuesta sin sustento**: respuesta que no puede respaldarse con ninguna Política; el asistente debe declarar que no tiene la información en lugar de inventarla.
