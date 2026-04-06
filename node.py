from comfy_api.latest import io
import node_helpers

class FluxReferenceConditioningNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image{n}" for n in range(1,11)],
            min=0,
        )
        return io.Schema(
            node_id="FluxReferenceConditioningNode",
            display_name="Flux Reference Conditioning",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Vae.Input("vae"),
                io.Autogrow.Input("images",template=image_template,tooltip="Images are available as image1-10"),
                ],
            outputs=[
                io.Conditioning.Output("conditioning"),
                ]
        )

    @classmethod
    def execute(cls, conditioning, vae, images: io.Autogrow.Type) -> io.NodeOutput:
        if 0 < len(images.values()):
            latent_list = [vae.encode(v) for v in images.values() if v is not None]
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": latent_list}, append=True)
        return io.NodeOutput(conditioning)

class FluxKleinReferenceConditioningNode(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        image_template = io.Autogrow.TemplateNames(
            io.Image.Input("image"),
            names=[f"image{n}" for n in range(1,11)],
            min=0,
        )
        return io.Schema(
            node_id="FluxKleinReferenceConditioningNode",
            display_name="Flux Klein Reference Conditioning",
            category="advanced/conditioning",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Autogrow.Input("images",template=image_template,tooltip="Images are available as image1-10"),
                ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                ]
        )

    @classmethod
    def execute(cls, positive, negative, vae, images: io.Autogrow.Type) -> io.NodeOutput:
        if 0 < len(images.values()):
            latent_list = [vae.encode(v) for v in images.values() if v is not None]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": latent_list}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": latent_list}, append=True)
        return io.NodeOutput(positive, negative)
