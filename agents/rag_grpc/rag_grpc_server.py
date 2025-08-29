import grpc
from concurrent import futures
import rag_grpc.rag_pb2 as rag_pb2
import rag_grpc.rag_pb2_grpc as rag_pb2_grpc
from RAG import ContentRetriever


class RagServiceServicer(rag_pb2_grpc.RagServiceServicer):
    def __init__(self):
        self.retriever = ContentRetriever()


    def GetRagS2C(self, request, context):
        docs = self.retriever.get_rag_from_summary2content(request.query, request.source)
        ctx = "\n\n".join(doc.page_content for doc in docs)
        return rag_pb2.RagReply(context=ctx)


    def GetRagContent(self, request, context):
        docs = self.retriever.get_rag_only_content(request.query)
        ctx = "\n\n".join(doc.page_content for doc in docs)
        return rag_pb2.RagReply(context=ctx)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    rag_pb2_grpc.add_RagServiceServicer_to_server(RagServiceServicer(), server)
    server.add_insecure_port('[::]:5413')
    server.start()
    print("RAG gRPC server started on port 5413")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
